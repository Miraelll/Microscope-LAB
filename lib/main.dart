import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  List<CameraDescription> cameras = [];
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Ошибка поиска камер: $e');
  }

  final CameraDescription? backCamera = cameras.isEmpty
      ? null
      : cameras.firstWhere(
          (c) => c.lensDirection == CameraLensDirection.back,
          orElse: () => cameras.first,
        );

  runApp(MicroscopeApp(camera: backCamera));
}

// Набор лабораторных оптических фильтров
enum LabFilter {
  standard('Стандарт'),
  highDetail('Контуры HD'),
  coldLight('Чистый белый'),
  achromat('Ахромат Ч/Б'),
  darkField('Темное поле');

  final String title;
  const LabFilter(this.title);
}

// Модель одной распознанной капсулы (координаты и радиус нормализованы к ширине кадра 0.0..1.0)
class DetectedCapsule {
  final double cxNorm; // X центра относительно ширины (0..1)
  final double cyNorm; // Y центра относительно высоты (0..1)
  final double rNorm; // Радиус относительно ширины (0..1)
  final double score;

  const DetectedCapsule({
    required this.cxNorm,
    required this.cyNorm,
    required this.rNorm,
    this.score = 1.0,
  });

  // Диаметр в пикселях для заданной ширины изображения
  double diameterPx(double imageWidth) => 2.0 * rNorm * imageWidth;

  // Диаметр в микрометрах (мкм), где калибровочный отрезок 0.20 * width равен scaleMicrons
  double diameterMicrons(double scaleMicrons) =>
      2.0 * rNorm * (scaleMicrons / 0.20);
}

// Входные данные для фонового изолята мульти-расчета капсул
class MultiPassTask {
  final Uint8List rgbaBytes;
  final int width;
  final int height;
  final List<double> brightnessList;
  final List<double> sharpnessList;
  final int filterIndex;
  final int minRadiusPx;
  final int maxRadiusPx;
  final double sensitivity; // 0.1 .. 1.0

  const MultiPassTask({
    required this.rgbaBytes,
    required this.width,
    required this.height,
    required this.brightnessList,
    required this.sharpnessList,
    required this.filterIndex,
    required this.minRadiusPx,
    required this.maxRadiusPx,
    required this.sensitivity,
  });
}

// Результат одного прохода при конкретной паре (Яркость, Четкость)
class SinglePassSummary {
  final double brightness;
  final double sharpness;
  final List<DetectedCapsule> capsules;

  const SinglePassSummary({
    required this.brightness,
    required this.sharpness,
    required this.capsules,
  });
}

// Итоговый результат мульти-расчета
class MultiPassResult {
  final List<DetectedCapsule> bestCapsules;
  final double averageCount;
  final int minCount;
  final int maxCount;
  final double bestBrightness;
  final double bestSharpness;
  final List<SinglePassSummary> passes;

  const MultiPassResult({
    required this.bestCapsules,
    required this.averageCount,
    required this.minCount,
    required this.maxCount,
    required this.bestBrightness,
    required this.bestSharpness,
    required this.passes,
  });
}

// ============================================================================
// ФОНОВЫЙ АЛГОРИТМ ПОИСКА КАПСУЛ (ПРЕОБРАЗОВАНИЕ ХАФА + ГРАДИЕНТНЫЙ АНАЛИЗ)
// Выполняется в отдельном потоке (Isolate), не блокируя интерфейс
// ============================================================================
MultiPassResult _runMultiPassIsolate(MultiPassTask task) {
  final List<SinglePassSummary> passes = [];

  for (final double b in task.brightnessList) {
    for (final double s in task.sharpnessList) {
      final capsules = _detectCapsulesSinglePass(
        rgba: task.rgbaBytes,
        w: task.width,
        h: task.height,
        brightness: b,
        sharpness: s,
        filterIndex: task.filterIndex,
        minR: task.minRadiusPx,
        maxR: task.maxRadiusPx,
        sensitivity: task.sensitivity,
      );
      passes.add(
        SinglePassSummary(brightness: b, sharpness: s, capsules: capsules),
      );
    }
  }

  if (passes.isEmpty) {
    return const MultiPassResult(
      bestCapsules: [],
      averageCount: 0,
      minCount: 0,
      maxCount: 0,
      bestBrightness: 0.0,
      bestSharpness: 1.0,
      passes: [],
    );
  }

  double total = 0;
  int minC = passes.first.capsules.length;
  int maxC = passes.first.capsules.length;

  for (final p in passes) {
    final c = p.capsules.length;
    total += c;
    if (c < minC) minC = c;
    if (c > maxC) maxC = c;
  }

  final double avgCount = total / passes.length;

  // Выбираем проход, количество капсул в котором ближе всего к среднему значению
  SinglePassSummary bestPass = passes.first;
  double minDiff = (bestPass.capsules.length - avgCount).abs();

  for (final p in passes) {
    final diff = (p.capsules.length - avgCount).abs();
    if (diff < minDiff) {
      minDiff = diff;
      bestPass = p;
    }
  }

  return MultiPassResult(
    bestCapsules: bestPass.capsules,
    averageCount: avgCount,
    minCount: minC,
    maxCount: maxC,
    bestBrightness: bestPass.brightness,
    bestSharpness: bestPass.sharpness,
    passes: passes,
  );
}

List<DetectedCapsule> _detectCapsulesSinglePass({
  required Uint8List rgba,
  required int w,
  required int h,
  required double brightness,
  required double sharpness,
  required int filterIndex,
  required int minR,
  required int maxR,
  required double sensitivity,
}) {
  final int numPixels = w * h;
  final Float32List lum = Float32List(numPixels);

  // 1. Применяем настройки Яркости, Четкости и выбранного фильтра к пикселям
  double contrast = 1.0 + (sharpness * 0.14);
  double brightnessOffset = brightness * 8.0;
  double rScale = 1.0, gScale = 1.0, bScale = 1.0;

  if (filterIndex == LabFilter.highDetail.index) {
    contrast *= 1.28;
    brightnessOffset -= 12.0;
  } else if (filterIndex == LabFilter.coldLight.index) {
    rScale = 0.92;
    gScale = 0.98;
    bScale = 1.14;
    contrast *= 1.08;
  } else if (filterIndex == LabFilter.achromat.index) {
    contrast *= 1.15;
  }

  final bool isInverted = filterIndex == LabFilter.darkField.index;

  for (int i = 0; i < numPixels; i++) {
    final int idx = i * 4;
    double r = rgba[idx].toDouble();
    double g = rgba[idx + 1].toDouble();
    double b = rgba[idx + 2].toDouble();

    if (isInverted) {
      r = 255.0 - r;
      g = 255.0 - g;
      b = 255.0 - b;
    }

    r = (r - 128.0) * (contrast * rScale) + 128.0 + brightnessOffset;
    g = (g - 128.0) * (contrast * gScale) + 128.0 + brightnessOffset;
    b = (b - 128.0) * (contrast * bScale) + 128.0 + brightnessOffset;

    r = r.clamp(0.0, 255.0);
    g = g.clamp(0.0, 255.0);
    b = b.clamp(0.0, 255.0);

    lum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }

  // 2. Сглаживание 3x3 для устранения мелкого шума матрицы
  final Float32List smooth = Float32List(numPixels);
  for (int y = 1; y < h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      final int p = y * w + x;
      smooth[p] =
          (lum[p - w - 1] +
              lum[p - w] * 2 +
              lum[p - w + 1] +
              lum[p - 1] * 2 +
              lum[p] * 4 +
              lum[p + 1] * 2 +
              lum[p + w - 1] +
              lum[p + w] * 2 +
              lum[p + w + 1]) /
          16.0;
    }
  }

  // 3. Вычисление градиентов Собеля (Gx, Gy и магнитуды G)
  final Float32List gxArr = Float32List(numPixels);
  final Float32List gyArr = Float32List(numPixels);
  final Float32List magArr = Float32List(numPixels);

  double sumMag = 0;
  int magCount = 0;

  for (int y = 2; y < h - 2; y++) {
    for (int x = 2; x < w - 2; x++) {
      final int p = y * w + x;
      final double gx =
          -smooth[p - w - 1] +
          smooth[p - w + 1] -
          2.0 * smooth[p - 1] +
          2.0 * smooth[p + 1] -
          smooth[p + w - 1] +
          smooth[p + w + 1];

      final double gy =
          -smooth[p - w - 1] -
          2.0 * smooth[p - w] -
          smooth[p - w + 1] +
          smooth[p + w - 1] +
          2.0 * smooth[p + w] +
          smooth[p + w + 1];

      final double mag = math.sqrt(gx * gx + gy * gy);
      gxArr[p] = gx;
      gyArr[p] = gy;
      magArr[p] = mag;

      if (mag > 12.0) {
        sumMag += mag;
        magCount++;
      }
    }
  }

  if (magCount < 20) return [];

  final double meanMag = sumMag / magCount;
  // Адаптивный порог границ с учетом чувствительности
  final double edgeThreshold = math.max(
    18.0,
    meanMag * (1.25 - sensitivity * 0.55),
  );

  // Список граничных пикселей для быстрого голосования Хафа
  final List<int> edgeIndices = [];
  for (int y = 4; y < h - 4; y++) {
    for (int x = 4; x < w - 4; x++) {
      final int p = y * w + x;
      if (magArr[p] >= edgeThreshold) {
        edgeIndices.add(p);
      }
    }
  }

  // 4. Круговое преобразование Хафа (Circle Hough Transform) по диапазону радиусов
  final int clampedMinR = minR.clamp(4, math.min(w, h) ~/ 3);
  final int clampedMaxR = maxR.clamp(clampedMinR + 2, math.min(w, h) ~/ 2);
  final int rStep = (clampedMaxR - clampedMinR) > 24 ? 2 : 1;

  final List<DetectedCapsule> rawCandidates = [];
  final Uint16List acc = Uint16List(numPixels);

  // Предвычисленные 16 точек на окружности для проверки радиальной целостности контура капсулы
  const int numAngles = 16;
  final List<double> cosTable = List.generate(
    numAngles,
    (i) => math.cos(i * 2 * math.pi / numAngles),
  );
  final List<double> sinTable = List.generate(
    numAngles,
    (i) => math.sin(i * 2 * math.pi / numAngles),
  );

  for (int r = clampedMinR; r <= clampedMaxR; r += rStep) {
    acc.fillRange(0, numPixels, 0);

    // Голосование вдоль направления градиента (+r и -r)
    for (int i = 0; i < edgeIndices.length; i++) {
      final int p = edgeIndices[i];
      final int x = p % w;
      final int y = p ~/ w;
      final double mag = magArr[p];
      final double ux = gxArr[p] / mag;
      final double uy = gyArr[p] / mag;

      final int cx1 = (x + ux * r).round();
      final int cy1 = (y + uy * r).round();
      if (cx1 >= r && cx1 < w - r && cy1 >= r && cy1 < h - r) {
        acc[cy1 * w + cx1] += 1;
      }

      final int cx2 = (x - ux * r).round();
      final int cy2 = (y - uy * r).round();
      if (cx2 >= r && cx2 < w - r && cy2 >= r && cy2 < h - r) {
        acc[cy2 * w + cx2] += 1;
      }
    }

    // Поиск локальных пиков аккумулятора и проверка круговой симметрии (защита от прямых линий)
    final int minVotes = math.max(
      5,
      (r * 0.45 * (1.3 - sensitivity * 0.5)).round(),
    );

    for (int cy = r; cy < h - r; cy += 2) {
      for (int cx = r; cx < w - r; cx += 2) {
        final int idx = cy * w + cx;
        final int votes = acc[idx];
        if (votes < minVotes) continue;

        // Проверяем, что это локальный максимум в окрестности 3x3
        if (votes < acc[idx - 1] ||
            votes < acc[idx + 1] ||
            votes < acc[idx - w] ||
            votes < acc[idx + w]) {
          continue;
        }

        // Проверяем, сколько секторов из 16 реально имеют границу на расстоянии r
        int activeSectors = 0;
        double ringGradSum = 0;
        for (int a = 0; a < numAngles; a++) {
          final int px = (cx + r * cosTable[a]).round();
          final int py = (cy + r * sinTable[a]).round();
          final double g = magArr[py * w + px];
          if (g >= edgeThreshold * 0.70) {
            activeSectors++;
            ringGradSum += g;
          }
        }

        // Требуем минимум 9 из 16 секторов (более 55% дуги окружности)
        final int requiredSectors = (10 - (sensitivity * 3).round()).clamp(
          7,
          12,
        );
        if (activeSectors >= requiredSectors) {
          final double score =
              (votes * 1.5) +
              (activeSectors * 4.0) +
              (ringGradSum / numAngles) * 0.1;
          rawCandidates.add(
            DetectedCapsule(
              cxNorm: cx / w,
              cyNorm: cy / h,
              rNorm: r / w,
              score: score,
            ),
          );
        }
      }
    }
  }

  // 5. Подавление немаксимумов (NMS) — убираем дублирующиеся и вложенные окружности
  rawCandidates.sort((a, b) => b.score.compareTo(a.score));
  final List<DetectedCapsule> accepted = [];
  final double aspect = h / w;

  for (final cand in rawCandidates) {
    bool overlaps = false;
    for (final ex in accepted) {
      final double dx = cand.cxNorm - ex.cxNorm;
      final double dy = (cand.cyNorm - ex.cyNorm) * aspect;
      final double dist = math.sqrt(dx * dx + dy * dy);
      final double minAllowedDist = math.max(cand.rNorm, ex.rNorm) * 0.80;
      if (dist < minAllowedDist) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      accepted.add(cand);
    }
  }

  return accepted;
}

// ============================================================================
// ОСНОВНОЕ ПРИЛОЖЕНИЕ
// ============================================================================
class MicroscopeApp extends StatelessWidget {
  final CameraDescription? camera;

  const MicroscopeApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(scaffoldBackgroundColor: Colors.black),
      home: CameraScreen(camera: camera),
    );
  }
}

class CameraScreen extends StatefulWidget {
  final CameraDescription? camera;

  const CameraScreen({super.key, required this.camera});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  bool _isCameraInitialized = false;
  final ImagePicker _picker = ImagePicker();

  // Источник изображения: null = режим живой камеры, иначе = загруженное/снятое фото
  Uint8List? _loadedImageBytes;
  int _loadedImageWidth = 1080;
  int _loadedImageHeight = 1920;

  // Параметры спинбоксов (шаг 0.1) — работают и для камеры, и для фото из галереи!
  double _brightness = 0.0;
  double _sharpness = 1.0;

  LabFilter _selectedFilter = LabFilter.standard;
  bool _isWide9x16 = true;

  double _minExposure = -2.0;
  double _maxExposure = 2.0;
  double _minZoom = 1.0;
  double _maxZoom = 5.0;
  double _currentZoom = 1.0;

  bool _isFocusLocked = false;
  Offset? _focusPoint;

  bool _showGrid = true;
  bool _useTimer = false;
  int _countdown = 0;
  bool _isProcessing = false;

  String _substanceName = 'Капсулы №1';
  String _magnification = '40x';
  List<String> _magnificationList = ['4x', '10x', '20x', '40x', '100x'];

  // Данные подсчета капсул и мульти-расчета
  List<DetectedCapsule> _capsules = [];
  double? _multiPassAverageCount;
  List<SinglePassSummary> _lastMultiPassDetails = [];
  bool _showCapsuleOverlay = true;
  bool _manualEditCapsules = false; // Ручное добавление/удаление капсул по тапу

  // Настройки алгоритма поиска и мульти-расчета
  String _multiBrightnessInput = '-0.5 0.0 0.5 1.0';
  String _multiSharpnessInput = '1.0 2.0 3.0';
  int _minCapsuleRadiusPx = 8;
  int _maxCapsuleRadiusPx = 45;
  double _detectionSensitivity = 0.65;

  @override
  void initState() {
    super.initState();
    _loadCustomMagnifications();
    if (widget.camera != null) {
      _initCamera();
    }
  }

  Future<void> _loadCustomMagnifications() async {
    final prefs = await SharedPreferences.getInstance();
    final savedList = prefs.getStringList('custom_magnifications');
    if (savedList != null && savedList.isNotEmpty) {
      setState(() {
        _magnificationList = savedList;
      });
    }
  }

  Future<void> _saveNewMagnification(String newVal) async {
    final formatted = newVal.endsWith('x') ? newVal : '${newVal}x';
    if (!_magnificationList.contains(formatted)) {
      setState(() {
        _magnificationList.add(formatted);
        _magnification = formatted;
      });
      final prefs = await SharedPreferences.getInstance();
      await prefs.setStringList('custom_magnifications', _magnificationList);
    } else {
      setState(() {
        _magnification = formatted;
      });
    }
  }

  Future<void> _initCamera() async {
    final cam = widget.camera;
    if (cam == null) return;

    final ctrl = CameraController(
      cam,
      ResolutionPreset.veryHigh,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    _controller = ctrl;

    try {
      await ctrl.initialize();
      await ctrl.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await ctrl.setFlashMode(FlashMode.off);

      try {
        _minExposure = await ctrl.getMinExposureOffset();
        _maxExposure = await ctrl.getMaxExposureOffset();
        _minZoom = await ctrl.getMinZoomLevel();
        _maxZoom = math.min(await ctrl.getMaxZoomLevel(), 8.0);
      } catch (_) {}

      if (mounted) {
        setState(() {
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      debugPrint('Ошибка инициализации камеры: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  // Калибровочная длина линейки (в мкм) для выбранного увеличения
  double _getScaleMicrons() {
    final int cleanMag =
        int.tryParse(_magnification.replaceAll(RegExp(r'[^0-9]'), '')) ?? 40;
    if (cleanMag <= 5) return 500.0;
    if (cleanMag <= 15) return 200.0;
    if (cleanMag <= 30) return 100.0;
    if (cleanMag <= 60) return 50.0;
    return 20.0;
  }

  // Статистика диаметров распознанных капсул (в мкм)
  Map<String, double> _getDiameterStatsMicrons() {
    if (_capsules.isEmpty) {
      return {'mean': 0.0, 'min': 0.0, 'max': 0.0};
    }
    final double scaleUm = _getScaleMicrons();
    double sum = 0.0;
    double minD = double.infinity;
    double maxD = 0.0;

    for (final c in _capsules) {
      final d = c.diameterMicrons(scaleUm);
      sum += d;
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;
    }

    return {'mean': sum / _capsules.length, 'min': minD, 'max': maxD};
  }

  ColorFilter _buildColorFilter({bool isFinalCapture = false}) {
    final double captureBoost = isFinalCapture ? 1.12 : 1.0;
    double contrast = (1.0 + (_sharpness * 0.14)) * captureBoost;
    double brightnessOffset = _brightness * 8.0;

    double rScale = 1.0;
    double gScale = 1.0;
    double bScale = 1.0;

    switch (_selectedFilter) {
      case LabFilter.standard:
        break;
      case LabFilter.highDetail:
        contrast *= 1.28;
        brightnessOffset -= 12.0;
        break;
      case LabFilter.coldLight:
        rScale = 0.92;
        gScale = 0.98;
        bScale = 1.14;
        contrast *= 1.08;
        break;
      case LabFilter.achromat:
        final double c = contrast * 1.15;
        final double t = 128.0 * (1.0 - c) + brightnessOffset;
        return ColorFilter.matrix(<double>[
          0.299 * c,
          0.587 * c,
          0.114 * c,
          0,
          t,
          0.299 * c,
          0.587 * c,
          0.114 * c,
          0,
          t,
          0.299 * c,
          0.587 * c,
          0.114 * c,
          0,
          t,
          0,
          0,
          0,
          1,
          0,
        ]);
      case LabFilter.darkField:
        final double c = -contrast;
        final double t = 255.0 + 128.0 * (1.0 - contrast) - brightnessOffset;
        return ColorFilter.matrix(<double>[
          c,
          0,
          0,
          0,
          t,
          0,
          c,
          0,
          0,
          t,
          0,
          0,
          c,
          0,
          t,
          0,
          0,
          0,
          1,
          0,
        ]);
    }

    final double tr = 128.0 * (1.0 - contrast * rScale) + brightnessOffset;
    final double tg = 128.0 * (1.0 - contrast * gScale) + brightnessOffset;
    final double tb = 128.0 * (1.0 - contrast * bScale) + brightnessOffset;

    return ColorFilter.matrix(<double>[
      contrast * rScale,
      0,
      0,
      0,
      tr,
      0,
      contrast * gScale,
      0,
      0,
      tg,
      0,
      0,
      contrast * bScale,
      0,
      tb,
      0,
      0,
      0,
      1,
      0,
    ]);
  }

  Future<void> _updateBrightness(double newValue) async {
    final clamped = newValue.clamp(-10.0, 10.0);
    final rounded = double.parse(clamped.toStringAsFixed(1));
    if (_brightness != rounded) {
      setState(() {
        _brightness = rounded;
      });
    }

    if (_loadedImageBytes == null && _controller != null) {
      try {
        final double norm = _brightness / 10.0;
        final double expOffset = norm >= 0
            ? norm * _maxExposure * 0.7
            : -norm * _minExposure * 0.7;
        await _controller!.setExposureOffset(
          expOffset.clamp(_minExposure, _maxExposure),
        );
      } catch (_) {}
    }
  }

  void _updateSharpness(double newValue) {
    final clamped = newValue.clamp(0.0, 10.0);
    final rounded = double.parse(clamped.toStringAsFixed(1));
    if (_sharpness != rounded) {
      setState(() {
        _sharpness = rounded;
      });
    }
  }

  Future<void> _updateZoom(double newValue) async {
    final clamped = newValue.clamp(_minZoom, _maxZoom);
    setState(() {
      _currentZoom = clamped;
    });
    if (_loadedImageBytes == null && _controller != null) {
      try {
        await _controller!.setZoomLevel(clamped);
      } catch (_) {}
    }
  }

  // ==========================================================================
  // ЗАГРУЗКА СУЩЕСТВУЮЩЕГО ФОТО ИЗ ГАЛЕРЕИ ИЛИ СЪЕМКА С КАМЕРЫ
  // ==========================================================================
  Future<void> _pickImageFromGallery() async {
    FocusManager.instance.primaryFocus?.unfocus();
    try {
      final XFile? picked = await _picker.pickImage(
        source: ImageSource.gallery,
      );
      if (picked == null) return;

      setState(() => _isProcessing = true);
      final Uint8List bytes = await File(picked.path).readAsBytes();
      await _setLoadedPhoto(bytes);
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Ошибка загрузки из галереи: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _setLoadedPhoto(Uint8List bytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo frame = await codec.getNextFrame();

    setState(() {
      _loadedImageBytes = bytes;
      _loadedImageWidth = frame.image.width;
      _loadedImageHeight = frame.image.height;
      _capsules = [];
      _multiPassAverageCount = null;
      _lastMultiPassDetails = [];
      _currentZoom = 1.0;
    });
  }

  // Возврат в режим живой камеры
  void _returnToLiveCamera() {
    FocusManager.instance.primaryFocus?.unfocus();
    setState(() {
      _loadedImageBytes = null;
      _capsules = [];
      _multiPassAverageCount = null;
      _lastMultiPassDetails = [];
      _manualEditCapsules = false;
    });
  }

  // Получение текущего кадра (либо загруженное фото, либо снимок с камеры)
  Future<Uint8List?> _obtainCurrentImageBytes() async {
    if (_loadedImageBytes != null) {
      return _loadedImageBytes;
    }
    if (_controller != null && _isCameraInitialized) {
      final XFile rawFile = await _controller!.takePicture();
      final Uint8List bytes = await File(rawFile.path).readAsBytes();
      await _setLoadedPhoto(bytes);
      return bytes;
    }
    return null;
  }

  // ==========================================================================
  // БЫСТРЫЙ ПОДСЧЕТ И МУЛЬТИ-РАСЧЕТ КАПСУЛ ПО НЕСКОЛЬКИМ ЯРКОСТЯМ И ЧЕТКОСТЯМ
  // ==========================================================================
  Future<void> _runCapsuleDetection({
    required List<double> brightnessValues,
    required List<double> sharpnessValues,
  }) async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing) return;
    setState(() => _isProcessing = true);

    try {
      final Uint8List? sourceBytes = await _obtainCurrentImageBytes();
      if (sourceBytes == null) return;

      // Масштабируем изображение до рабочей ширины 320 пикс для быстрого преобразования Хафа
      const int workWidth = 320;
      final int workHeight =
          ((workWidth * _loadedImageHeight) / _loadedImageWidth).round().clamp(
            180,
            640,
          );

      final ui.Codec codec = await ui.instantiateImageCodec(
        sourceBytes,
        targetWidth: workWidth,
        targetHeight: workHeight,
      );
      final ui.FrameInfo frame = await codec.getNextFrame();
      final ByteData? rgbaData = await frame.image.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );

      if (rgbaData == null) return;

      final task = MultiPassTask(
        rgbaBytes: Uint8List.fromList(rgbaData.buffer.asUint8List()),
        width: frame.image.width,
        height: frame.image.height,
        brightnessList: brightnessValues,
        sharpnessList: sharpnessValues,
        filterIndex: _selectedFilter.index,
        minRadiusPx: _minCapsuleRadiusPx,
        maxRadiusPx: _maxCapsuleRadiusPx,
        sensitivity: _detectionSensitivity,
      );

      // Запускаем распознавание в фоновом потоке (Isolate)
      final MultiPassResult result = await compute(_runMultiPassIsolate, task);

      if (!mounted) return;
      setState(() {
        _capsules = List<DetectedCapsule>.from(result.bestCapsules);
        _multiPassAverageCount = result.averageCount;
        _lastMultiPassDetails = result.passes;
        _showCapsuleOverlay = true;
        // Применяем оптимальную пару (Яркость, Четкость), давшую самый близкий к среднему результат
        _brightness = result.bestBrightness;
        _sharpness = result.bestSharpness;
      });

      final stats = _getDiameterStatsMicrons();
      final bool isMulti = result.passes.length > 1;

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          duration: const Duration(seconds: 3),
          backgroundColor: Colors.teal.shade800,
          content: Text(
            isMulti
                ? 'Мульти-расчет (${result.passes.length} проходов): Среднее = ${result.averageCount.toStringAsFixed(1)} шт. '
                      '(выбрано: ${_capsules.length} шт.) | Ср. диаметр: ${stats['mean']!.toStringAsFixed(1)} мкм'
                : 'Найдено капсул: ${_capsules.length} шт. | Ср. диаметр: ${stats['mean']!.toStringAsFixed(1)} мкм',
          ),
        ),
      );
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Ошибка подсчета капсул: $e')));
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  // Парсинг строки значений через пробел (как в CapsuleCounter)
  List<double> _parseSpaceSeparatedNumbers(
    String input,
    double minVal,
    double maxVal,
    double fallback,
  ) {
    final parts = input
        .replaceAll(',', '.')
        .split(RegExp(r'\s+'))
        .where((s) => s.trim().isNotEmpty);
    final List<double> res = [];
    for (final p in parts) {
      final v = double.tryParse(p);
      if (v != null) {
        res.add(double.parse(v.clamp(minVal, maxVal).toStringAsFixed(1)));
      }
    }
    return res.isEmpty ? [fallback] : res;
  }

  // Диалог настройки Мульти-расчета (ввод нескольких яркостей и четкостей + просмотр отчета)
  Future<void> _openMultiPassDialog() async {
    FocusManager.instance.primaryFocus?.unfocus();
    final TextEditingController bCtrl = TextEditingController(
      text: _multiBrightnessInput,
    );
    final TextEditingController sCtrl = TextEditingController(
      text: _multiSharpnessInput,
    );

    int tempMinR = _minCapsuleRadiusPx;
    int tempMaxR = _maxCapsuleRadiusPx;
    double tempSens = _detectionSensitivity;

    final bool? shouldRun = await showDialog<bool>(
      context: context,
      builder: (context) => StatefulBuilder(
        builder: (context, setDialogState) {
          return AlertDialog(
            title: const Text(
              'Мульти-расчет капсул',
              style: TextStyle(fontSize: 17),
            ),
            content: SingleChildScrollView(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Введите несколько значений через пробел. Программа рассчитает количество капсул для каждой комбинации и найдет среднее:',
                    style: TextStyle(fontSize: 12, color: Colors.white70),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: bCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Значения Яркости (через пробел)',
                      hintText: '-1.0 0.0 0.5 1.5',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: sCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Значения Четкости (через пробел)',
                      hintText: '1.0 2.0 3.0',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Align(
                    alignment: Alignment.centerRight,
                    child: TextButton.icon(
                      onPressed: () {
                        final b1 = (_brightness - 0.5)
                            .clamp(-10.0, 10.0)
                            .toStringAsFixed(1);
                        final b2 = _brightness.toStringAsFixed(1);
                        final b3 = (_brightness + 0.5)
                            .clamp(-10.0, 10.0)
                            .toStringAsFixed(1);

                        final s1 = (_sharpness - 0.5)
                            .clamp(0.0, 10.0)
                            .toStringAsFixed(1);
                        final s2 = _sharpness.toStringAsFixed(1);
                        final s3 = (_sharpness + 0.5)
                            .clamp(0.0, 10.0)
                            .toStringAsFixed(1);

                        setDialogState(() {
                          bCtrl.text = '$b1 $b2 $b3';
                          sCtrl.text = '$s1 $s2 $s3';
                        });
                      },
                      icon: const Icon(
                        Icons.grid_view,
                        size: 15,
                        color: Colors.cyanAccent,
                      ),
                      label: const Text(
                        'Заполнить вокруг текущих (±0.5)',
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.cyanAccent,
                        ),
                      ),
                    ),
                  ),
                  const Divider(),
                  Text(
                    'Радиус капсул на матрице: $tempMinR – $tempMaxR пикс.',
                    style: const TextStyle(fontSize: 12),
                  ),
                  RangeSlider(
                    values: RangeValues(
                      tempMinR.toDouble(),
                      tempMaxR.toDouble(),
                    ),
                    min: 4,
                    max: 90,
                    divisions: 86,
                    activeColor: Colors.cyanAccent,
                    labels: RangeLabels('$tempMinR px', '$tempMaxR px'),
                    onChanged: (vals) {
                      setDialogState(() {
                        tempMinR = vals.start.round();
                        tempMaxR = math.max(tempMinR + 2, vals.end.round());
                      });
                    },
                  ),
                  Text(
                    'Чувствительность контуров: ${(tempSens * 100).round()}%',
                    style: const TextStyle(fontSize: 12),
                  ),
                  Slider(
                    value: tempSens,
                    min: 0.2,
                    max: 0.95,
                    activeColor: Colors.amberAccent,
                    onChanged: (val) => setDialogState(() => tempSens = val),
                  ),
                  if (_lastMultiPassDetails.isNotEmpty) ...[
                    const Divider(),
                    Text(
                      'Последний мульти-расчет (Среднее: ${_multiPassAverageCount?.toStringAsFixed(1)} шт.):',
                      style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.cyanAccent,
                      ),
                    ),
                    const SizedBox(height: 4),
                    ConstrainedBox(
                      constraints: const BoxConstraints(maxHeight: 100),
                      child: ListView.builder(
                        shrinkWrap: true,
                        itemCount: _lastMultiPassDetails.length,
                        itemBuilder: (context, idx) {
                          final p = _lastMultiPassDetails[idx];
                          return Text(
                            '• Ярк: ${p.brightness.toStringAsFixed(1)} | Четк: ${p.sharpness.toStringAsFixed(1)}  →  ${p.capsules.length} капсул',
                            style: const TextStyle(
                              fontSize: 11,
                              color: Colors.white70,
                            ),
                          );
                        },
                      ),
                    ),
                  ],
                ],
              ),
            ),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context, false),
                child: const Text('Отмена'),
              ),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.cyanAccent,
                  foregroundColor: Colors.black,
                ),
                onPressed: () => Navigator.pop(context, true),
                icon: const Icon(Icons.analytics, size: 18),
                label: const Text('Рассчитать среднее'),
              ),
            ],
          );
        },
      ),
    );

    if (shouldRun == true) {
      _multiBrightnessInput = bCtrl.text.trim();
      _multiSharpnessInput = sCtrl.text.trim();
      _minCapsuleRadiusPx = tempMinR;
      _maxCapsuleRadiusPx = tempMaxR;
      _detectionSensitivity = tempSens;

      final bList = _parseSpaceSeparatedNumbers(
        _multiBrightnessInput,
        -10.0,
        10.0,
        _brightness,
      );
      final sList = _parseSpaceSeparatedNumbers(
        _multiSharpnessInput,
        0.0,
        10.0,
        _sharpness,
      );

      await _runCapsuleDetection(
        brightnessValues: bList,
        sharpnessValues: sList,
      );
    }
  }

  // АВТОНАСТРОЙКА яркости и четкости (работает и для камеры, и для загруженного фото!)
  Future<void> _autoTuneImage() async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing) return;
    setState(() => _isProcessing = true);

    try {
      Uint8List? bytes = _loadedImageBytes;
      if (bytes == null && _controller != null && _isCameraInitialized) {
        await _controller!.setExposureOffset(0.0);
        final XFile sampleFile = await _controller!.takePicture();
        bytes = await File(sampleFile.path).readAsBytes();
      }
      if (bytes == null) return;

      final ui.Codec codec = await ui.instantiateImageCodec(
        bytes,
        targetWidth: 200,
        targetHeight: 200,
      );
      final ui.FrameInfo frame = await codec.getNextFrame();
      final ByteData? rgbaData = await frame.image.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );

      if (rgbaData != null) {
        final Uint8List pixels = rgbaData.buffer.asUint8List();
        final int width = frame.image.width;
        final int height = frame.image.height;

        double sumLum = 0;
        double sumLumSq = 0;
        int count = 0;

        final double cx = width / 2.0;
        final double cy = height / 2.0;
        final double maxRadiusSq = math
            .pow(math.min(width, height) * 0.38, 2)
            .toDouble();

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final double dx = x - cx;
            final double dy = y - cy;
            if (dx * dx + dy * dy <= maxRadiusSq) {
              final int idx = (y * width + x) * 4;
              final int r = pixels[idx];
              final int g = pixels[idx + 1];
              final int b = pixels[idx + 2];
              final double lum = 0.299 * r + 0.587 * g + 0.114 * b;

              if (lum > 15) {
                sumLum += lum;
                sumLumSq += lum * lum;
                count++;
              }
            }
          }
        }

        if (count > 50) {
          final double meanLum = sumLum / count;
          final double variance = (sumLumSq / count) - (meanLum * meanLum);
          final double stdDev = math.sqrt(math.max(0, variance));

          final double targetBrightness = ((135.0 - meanLum) / 18.0).clamp(
            -6.0,
            6.0,
          );
          final double targetSharpness = ((55.0 - stdDev) / 8.0).clamp(
            1.0,
            7.0,
          );

          await _updateBrightness(targetBrightness);
          _updateSharpness(targetSharpness);

          if (mounted) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                duration: const Duration(seconds: 2),
                content: Text(
                  'Авто: Яркость ${_brightness.toStringAsFixed(1)}, Четкость ${_sharpness.toStringAsFixed(1)}',
                ),
              ),
            );
          }
        }
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(SnackBar(content: Text('Ошибка автонастройки: $e')));
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _toggleFocusLock() async {
    if (_controller == null || !_isCameraInitialized) return;
    try {
      final newLockState = !_isFocusLocked;
      await _controller!.setFocusMode(
        newLockState ? FocusMode.locked : FocusMode.auto,
      );
      setState(() {
        _isFocusLocked = newLockState;
      });
    } catch (_) {}
  }

  // Нажатие по экрану: либо ручное добавление/удаление капсулы, либо фокусировка камеры
  Future<void> _onTapViewport(TapDownDetails details, Size viewSize) async {
    if (FocusScope.of(context).hasFocus) {
      FocusScope.of(context).unfocus();
      return;
    }

    final double tapXNorm = (details.localPosition.dx / viewSize.width).clamp(
      0.0,
      1.0,
    );
    final double tapYNorm = (details.localPosition.dy / viewSize.height).clamp(
      0.0,
      1.0,
    );

    // Если включен режим ручной правки капсул на снимке
    if (_manualEditCapsules && _loadedImageBytes != null) {
      final double aspect = viewSize.height / viewSize.width;
      int hitIndex = -1;

      for (int i = 0; i < _capsules.length; i++) {
        final c = _capsules[i];
        final dx = tapXNorm - c.cxNorm;
        final dy = (tapYNorm - c.cyNorm) * aspect;
        if (math.sqrt(dx * dx + dy * dy) <= c.rNorm * 1.15) {
          hitIndex = i;
          break;
        }
      }

      setState(() {
        if (hitIndex >= 0) {
          // Удаляем лишнюю капсулу при нажатии на неё
          _capsules.removeAt(hitIndex);
        } else {
          // Добавляем капсулу со средним радиусом
          double defaultRNorm = 0.04;
          if (_capsules.isNotEmpty) {
            defaultRNorm =
                _capsules.map((e) => e.rNorm).reduce((a, b) => a + b) /
                _capsules.length;
          }
          _capsules.add(
            DetectedCapsule(
              cxNorm: tapXNorm,
              cyNorm: tapYNorm,
              rNorm: defaultRNorm,
            ),
          );
        }
      });
      return;
    }

    // Иначе — фокусировка живой камеры
    if (_loadedImageBytes == null &&
        _controller != null &&
        _isCameraInitialized) {
      setState(() {
        _focusPoint = details.localPosition;
      });

      try {
        final offset = Offset(tapXNorm, tapYNorm);
        await _controller!.setFocusPoint(offset);
        await _controller!.setExposurePoint(offset);
        await _controller!.setFocusMode(FocusMode.locked);
        setState(() {
          _isFocusLocked = true;
        });
      } catch (_) {}

      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) {
          setState(() {
            _focusPoint = null;
          });
        }
      });
    }
  }

  Future<void> _editSubstanceDialog() async {
    FocusManager.instance.primaryFocus?.unfocus();
    final TextEditingController textCtrl = TextEditingController(
      text: _substanceName,
    );
    final String? result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Подпись образца на снимке'),
        content: TextField(
          controller: textCtrl,
          autofocus: true,
          decoration: const InputDecoration(
            labelText: 'Название образца / капсул',
            hintText: 'Например: Микрокапсулы альгината',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Отмена'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, textCtrl.text.trim()),
            child: const Text('Сохранить'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      setState(() {
        _substanceName = result;
      });
    }
  }

  Future<void> _editMagnificationDialog() async {
    FocusManager.instance.primaryFocus?.unfocus();
    final TextEditingController customCtrl = TextEditingController();

    final String? result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Выберите или введите увеличение'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: _magnificationList.map((val) {
                return ActionChip(
                  label: Text(
                    val,
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  backgroundColor: _magnification == val
                      ? Colors.cyanAccent
                      : Colors.white12,
                  labelStyle: TextStyle(
                    color: _magnification == val ? Colors.black : Colors.white,
                  ),
                  onPressed: () => Navigator.pop(context, val),
                );
              }).toList(),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: customCtrl,
              decoration: const InputDecoration(
                labelText: 'Свое значение (например: 60x)',
                border: OutlineInputBorder(),
              ),
              onSubmitted: (val) {
                if (val.trim().isNotEmpty) {
                  Navigator.pop(context, val.trim());
                }
              },
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Отмена'),
          ),
          ElevatedButton(
            onPressed: () {
              final txt = customCtrl.text.trim();
              if (txt.isNotEmpty) {
                Navigator.pop(context, txt);
              } else {
                Navigator.pop(context);
              }
            },
            child: const Text('Сохранить и выбрать'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      await _saveNewMagnification(result);
    }
  }

  // Сохранение итогового изображения (с камеры или отредактированного фото из галереи)
  Future<void> _captureOrSaveImage() async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing) return;

    if (_loadedImageBytes == null && _useTimer) {
      for (int i = 3; i > 0; i--) {
        if (!mounted) return;
        setState(() => _countdown = i);
        await Future.delayed(const Duration(seconds: 1));
      }
      if (!mounted) return;
      setState(() => _countdown = 0);
    }

    setState(() => _isProcessing = true);

    try {
      final Uint8List? rawBytes = await _obtainCurrentImageBytes();
      if (rawBytes == null) return;

      final Uint8List processedBytes = await _processAndStampImage(rawBytes);

      final Directory tempDir = await getTemporaryDirectory();
      final String outPath =
          '${tempDir.path}/capsules_${DateTime.now().millisecondsSinceEpoch}.png';
      final File outFile = File(outPath);
      await outFile.writeAsBytes(processedBytes);

      final bool hasAccess = await Gal.hasAccess();
      if (!hasAccess) {
        await Gal.requestAccess();
      }
      await Gal.putImage(outFile.path);

      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: Colors.green.shade800,
          content: Text('Сохранено в галерею! Капсул: ${_capsules.length} шт.'),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('Ошибка сохранения: $e')));
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  // Отрисовка фильтров, распознанных капсул, их диаметров и лабораторного штампа
  Future<Uint8List> _processAndStampImage(Uint8List inputBytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(inputBytes);
    final ui.FrameInfo frame = await codec.getNextFrame();
    final ui.Image srcImage = frame.image;

    final int w = srcImage.width;
    final int h = srcImage.height;

    final ui.PictureRecorder recorder = ui.PictureRecorder();
    final Canvas canvas = Canvas(
      recorder,
      Rect.fromLTWH(0, 0, w.toDouble(), h.toDouble()),
    );

    // 1. Отрисовка снимка с выбранной яркостью, четкостью и оптическим фильтром
    final Paint imagePaint = Paint()
      ..colorFilter = _buildColorFilter(isFinalCapture: true)
      ..filterQuality = FilterQuality.high;
    canvas.drawImage(srcImage, Offset.zero, imagePaint);

    final double scaleUm = _getScaleMicrons();

    // 2. Отрисовка распознанных капсул и их диаметров на снимке
    if (_showCapsuleOverlay && _capsules.isNotEmpty) {
      final Paint circlePaint = Paint()
        ..color = Colors.greenAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.max(2.5, w * 0.0035);

      final Paint centerPaint = Paint()
        ..color = Colors.amberAccent
        ..style = PaintingStyle.fill;

      for (int i = 0; i < _capsules.length; i++) {
        final c = _capsules[i];
        final double cx = c.cxNorm * w;
        final double cy = c.cyNorm * h;
        final double r = c.rNorm * w;
        final double diamUm = c.diameterMicrons(scaleUm);

        canvas.drawCircle(Offset(cx, cy), r, circlePaint);
        canvas.drawCircle(
          Offset(cx, cy),
          math.max(2.0, w * 0.004),
          centerPaint,
        );

        final TextPainter tp = TextPainter(
          text: TextSpan(
            text: '#${i + 1} (${diamUm.toStringAsFixed(0)} мкм)',
            style: TextStyle(
              color: Colors.yellowAccent,
              fontSize: math.max(11.0, w * 0.020),
              fontWeight: FontWeight.bold,
              backgroundColor: const Color.fromRGBO(0, 0, 0, 0.65),
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();

        tp.paint(canvas, Offset(cx - tp.width / 2, cy - r - tp.height - 2));
      }
    }

    // 3. Формирование 4 строк лабораторного штампа (включая подсчет капсул и средний диаметр)
    final DateTime now = DateTime.now();
    final String dateStr =
        '${now.day.toString().padLeft(2, '0')}.${now.month.toString().padLeft(2, '0')}.${now.year} '
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';

    final stats = _getDiameterStatsMicrons();
    final String avgText = _multiPassAverageCount != null
        ? ' (Ср. мульти: ${_multiPassAverageCount!.toStringAsFixed(1)})'
        : '';

    final String line1Title = '$_substanceName  |  Увеличение: $_magnification';
    final String line2Capsules = _capsules.isNotEmpty
        ? 'Капсул: ${_capsules.length} шт.$avgText  |  Ср. диаметр: ${stats['mean']!.toStringAsFixed(1)} мкм [${stats['min']!.toStringAsFixed(0)}–${stats['max']!.toStringAsFixed(0)} мкм]'
        : 'Капсулы: подсчет не проводился';
    final String line3Params =
        'Фильтр: ${_selectedFilter.title}  |  Ярк: ${_brightness.toStringAsFixed(1)}  |  Четк: ${_sharpness.toStringAsFixed(1)}';
    final String line4Date = 'Дата съемки: $dateStr';

    final double textMaxWidth = w * 0.70;
    final double titleFontSize = w * 0.033;
    final double subFontSize = w * 0.023;

    TextPainter buildPainter(
      String text,
      double size,
      Color color, {
      bool bold = false,
    }) {
      return TextPainter(
        text: TextSpan(
          text: text,
          style: TextStyle(
            color: color,
            fontSize: size,
            fontWeight: bold ? FontWeight.bold : FontWeight.normal,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout(maxWidth: textMaxWidth);
    }

    final tp1 = buildPainter(
      line1Title,
      titleFontSize,
      Colors.white,
      bold: true,
    );
    final tp2 = buildPainter(
      line2Capsules,
      subFontSize * 1.05,
      Colors.greenAccent,
      bold: true,
    );
    final tp3 = buildPainter(line3Params, subFontSize, Colors.white70);
    final tp4 = buildPainter(line4Date, subFontSize * 0.92, Colors.white54);

    final double verticalPadding = h * 0.015;
    final double lineSpacing = h * 0.005;
    final double totalTextHeight =
        tp1.height +
        tp2.height +
        tp3.height +
        tp4.height +
        (lineSpacing * 3) +
        (verticalPadding * 2.2);

    final double barHeight = math.max(h * 0.115, totalTextHeight);
    final double barTop = h - barHeight;

    canvas.drawRect(
      Rect.fromLTWH(0, barTop, w.toDouble(), barHeight),
      Paint()..color = const Color.fromRGBO(0, 0, 0, 0.82),
    );

    double currentY = barTop + verticalPadding;
    tp1.paint(canvas, Offset(w * 0.03, currentY));
    currentY += tp1.height + lineSpacing;

    tp2.paint(canvas, Offset(w * 0.03, currentY));
    currentY += tp2.height + lineSpacing;

    tp3.paint(canvas, Offset(w * 0.03, currentY));
    currentY += tp3.height + lineSpacing;

    tp4.paint(canvas, Offset(w * 0.03, currentY));

    // 4. Калибровочная линейка в правом нижнем углу
    final String scaleLabel = '${scaleUm.toInt()} мкм';
    final double scaleBarWidth = w * 0.20;
    final double scaleRight = w * 0.96;
    final double scaleLeft = scaleRight - scaleBarWidth;
    final double scaleY = barTop + (barHeight * 0.65);

    final Paint scalePaint = Paint()
      ..color = Colors.cyanAccent
      ..strokeWidth = math.max(3.0, h * 0.003);

    canvas.drawLine(
      Offset(scaleLeft, scaleY),
      Offset(scaleRight, scaleY),
      scalePaint,
    );
    canvas.drawLine(
      Offset(scaleLeft, scaleY - barHeight * 0.09),
      Offset(scaleLeft, scaleY + barHeight * 0.09),
      scalePaint,
    );
    canvas.drawLine(
      Offset(scaleRight, scaleY - barHeight * 0.09),
      Offset(scaleRight, scaleY + barHeight * 0.09),
      scalePaint,
    );

    final TextPainter scaleTextPainter = TextPainter(
      text: TextSpan(
        text: scaleLabel,
        style: TextStyle(
          color: Colors.cyanAccent,
          fontSize: titleFontSize * 0.82,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    scaleTextPainter.paint(
      canvas,
      Offset(
        scaleLeft + (scaleBarWidth - scaleTextPainter.width) / 2,
        scaleY - scaleTextPainter.height - (barHeight * 0.11),
      ),
    );

    final ui.Picture picture = recorder.endRecording();
    final ui.Image finalImage = await picture.toImage(w, h);
    final ByteData? pngBytes = await finalImage.toByteData(
      format: ui.ImageByteFormat.png,
    );

    return pngBytes!.buffer.asUint8List();
  }

  @override
  Widget build(BuildContext context) {
    final bool isPhotoMode = _loadedImageBytes != null;
    final diameterStats = _getDiameterStatsMicrons();
    final double scaleUm = _getScaleMicrons();

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Column(
          children: [
            // 1. ВЕРХНЯЯ ПАНЕЛЬ (Название, Увеличение, Галерея, Сетка, Таймер)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 4),
              color: Colors.black87,
              child: Row(
                children: [
                  Expanded(
                    child: InkWell(
                      onTap: _editSubstanceDialog,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white10,
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: Colors.white24),
                        ),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.science,
                              size: 15,
                              color: Colors.cyanAccent,
                            ),
                            const SizedBox(width: 4),
                            Expanded(
                              child: Text(
                                _substanceName,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(fontSize: 12),
                              ),
                            ),
                            const Icon(
                              Icons.edit,
                              size: 12,
                              color: Colors.white54,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 5),
                  // Выбор/ввод увеличения
                  InkWell(
                    onTap: _editMagnificationDialog,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 7,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white10,
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(
                          color: Colors.amberAccent.withOpacity(0.5),
                        ),
                      ),
                      child: Row(
                        children: [
                          Text(
                            _magnification,
                            style: const TextStyle(
                              color: Colors.amberAccent,
                              fontWeight: FontWeight.bold,
                              fontSize: 12,
                            ),
                          ),
                          const Icon(
                            Icons.arrow_drop_down,
                            color: Colors.amberAccent,
                            size: 16,
                          ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(width: 4),
                  // Кнопка загрузки существующего фото из галереи
                  IconButton(
                    visualDensity: VisualDensity.compact,
                    tooltip: 'Загрузить фото из галереи',
                    icon: const Icon(
                      Icons.photo_library,
                      size: 20,
                      color: Colors.cyanAccent,
                    ),
                    onPressed: _pickImageFromGallery,
                  ),
                  // Если открыто фото — кнопка возврата к живой камере
                  if (isPhotoMode)
                    IconButton(
                      visualDensity: VisualDensity.compact,
                      tooltip: 'Вернуться к живой камере',
                      icon: const Icon(
                        Icons.videocam,
                        size: 22,
                        color: Colors.greenAccent,
                      ),
                      onPressed: _returnToLiveCamera,
                    ),
                  IconButton(
                    visualDensity: VisualDensity.compact,
                    icon: Icon(
                      Icons.grid_on,
                      size: 19,
                      color: _showGrid ? Colors.cyanAccent : Colors.white38,
                    ),
                    onPressed: () => setState(() => _showGrid = !_showGrid),
                  ),
                ],
              ),
            ),

            // 2. ПАНЕЛЬ СТАТИСТИКИ КАПСУЛ И ДИАМЕТРОВ
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              color: Colors.grey.shade900,
              child: Row(
                children: [
                  const Icon(
                    Icons.bubble_chart,
                    size: 16,
                    color: Colors.greenAccent,
                  ),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      _capsules.isEmpty
                          ? (isPhotoMode
                                ? 'Фото загружено. Нажмите «Считать» или «Мульти» для поиска капсул'
                                : 'Режим камеры (нажмите «Галерея» или «Считать»)')
                          : 'Капсул: ${_capsules.length} шт.${_multiPassAverageCount != null ? " (Ср: ${_multiPassAverageCount!.toStringAsFixed(1)})" : ""}  |  '
                                'Ø ср: ${diameterStats['mean']!.toStringAsFixed(1)} мкм '
                                '[${diameterStats['min']!.toStringAsFixed(0)}–${diameterStats['max']!.toStringAsFixed(0)}]',
                      style: TextStyle(
                        fontSize: 11.5,
                        fontWeight: FontWeight.w600,
                        color: _capsules.isEmpty
                            ? Colors.white70
                            : Colors.greenAccent,
                      ),
                    ),
                  ),
                  if (isPhotoMode) ...[
                    // Переключатель ручного добавления/удаления капсул пальцем
                    InkWell(
                      onTap: () => setState(
                        () => _manualEditCapsules = !_manualEditCapsules,
                      ),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 2,
                        ),
                        decoration: BoxDecoration(
                          color: _manualEditCapsules
                              ? Colors.amberAccent
                              : Colors.white12,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          _manualEditCapsules ? 'Правка: ВКЛ' : 'Правка',
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            color: _manualEditCapsules
                                ? Colors.black
                                : Colors.white70,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 6),
                    InkWell(
                      onTap: () => setState(
                        () => _showCapsuleOverlay = !_showCapsuleOverlay,
                      ),
                      child: Icon(
                        _showCapsuleOverlay
                            ? Icons.visibility
                            : Icons.visibility_off,
                        size: 16,
                        color: Colors.white70,
                      ),
                    ),
                  ],
                ],
              ),
            ),

            // 3. ОБЛАСТЬ ИЗОБРАЖЕНИЯ (Живая камера ИЛИ загруженное фото из галереи)
            Expanded(
              child: Center(
                child: LayoutBuilder(
                  builder: (context, outerConstraints) {
                    final double viewWidth = outerConstraints.maxWidth;
                    final double viewHeight = _isWide9x16
                        ? outerConstraints.maxHeight
                        : math.min(
                            outerConstraints.maxHeight,
                            viewWidth * (4.0 / 3.0),
                          );

                    return SizedBox(
                      width: viewWidth,
                      height: viewHeight,
                      child: ClipRect(
                        child: GestureDetector(
                          onTapDown: (details) => _onTapViewport(
                            details,
                            Size(viewWidth, viewHeight),
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              // A) Режим редактирования и анализа загруженного фото
                              if (isPhotoMode)
                                Transform.scale(
                                  scale: _currentZoom,
                                  child: SizedBox(
                                    width: viewWidth,
                                    height: viewHeight,
                                    child: ColorFiltered(
                                      colorFilter: _buildColorFilter(),
                                      child: Image.memory(
                                        _loadedImageBytes!,
                                        fit: BoxFit.fill,
                                        gaplessPlayback: true,
                                      ),
                                    ),
                                  ),
                                )
                              // B) Режим живой камеры
                              else if (_isCameraInitialized &&
                                  _controller != null)
                                SizedBox.expand(
                                  child: FittedBox(
                                    fit: BoxFit.cover,
                                    child: SizedBox(
                                      width: viewWidth,
                                      height:
                                          viewWidth *
                                          _controller!.value.aspectRatio,
                                      child: ColorFiltered(
                                        colorFilter: _buildColorFilter(),
                                        child: CameraPreview(_controller!),
                                      ),
                                    ),
                                  ),
                                )
                              else
                                Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const CircularProgressIndicator(),
                                    const SizedBox(height: 12),
                                    ElevatedButton.icon(
                                      onPressed: _pickImageFromGallery,
                                      icon: const Icon(Icons.photo_library),
                                      label: const Text(
                                        'Открыть фото из галереи',
                                      ),
                                    ),
                                  ],
                                ),

                              // Слой отрисовки распознанных капсул и их диаметров
                              if (_showCapsuleOverlay && _capsules.isNotEmpty)
                                IgnorePointer(
                                  child: CustomPaint(
                                    size: Size(viewWidth, viewHeight),
                                    painter: CapsuleOverlayPainter(
                                      capsules: _capsules,
                                      scaleMicrons: scaleUm,
                                    ),
                                  ),
                                ),

                              // Прицельная лабораторная сетка
                              if (_showGrid)
                                IgnorePointer(
                                  child: CustomPaint(
                                    size: Size(viewWidth, viewHeight),
                                    painter: MicroscopeGridPainter(),
                                  ),
                                ),

                              if (_focusPoint != null)
                                Positioned(
                                  left: _focusPoint!.dx - 25,
                                  top: _focusPoint!.dy - 25,
                                  child: Container(
                                    width: 50,
                                    height: 50,
                                    decoration: BoxDecoration(
                                      border: Border.all(
                                        color: Colors.greenAccent,
                                        width: 2,
                                      ),
                                      borderRadius: BorderRadius.circular(4),
                                    ),
                                  ),
                                ),

                              if (_countdown > 0)
                                Text(
                                  '$_countdown',
                                  style: const TextStyle(
                                    fontSize: 90,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.orangeAccent,
                                  ),
                                ),
                            ],
                          ),
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),

            // 4. НИЖНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ (Фильтры, Зум, Спинбоксы, Подсчет капсул)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              color: Colors.black87,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Оптические фильтры (работают и на камере, и на загруженном фото!)
                  SizedBox(
                    height: 28,
                    child: ListView.separated(
                      scrollDirection: Axis.horizontal,
                      itemCount: LabFilter.values.length,
                      separatorBuilder: (_, __) => const SizedBox(width: 6),
                      itemBuilder: (context, index) {
                        final filter = LabFilter.values[index];
                        final isSelected = filter == _selectedFilter;
                        return ChoiceChip(
                          label: Text(
                            filter.title,
                            style: TextStyle(
                              fontSize: 11,
                              fontWeight: isSelected
                                  ? FontWeight.bold
                                  : FontWeight.normal,
                              color: isSelected ? Colors.black : Colors.white70,
                            ),
                          ),
                          selected: isSelected,
                          selectedColor: Colors.cyanAccent,
                          backgroundColor: Colors.white10,
                          visualDensity: VisualDensity.compact,
                          padding: const EdgeInsets.symmetric(horizontal: 4),
                          showCheckmark: false,
                          onSelected: (_) =>
                              setState(() => _selectedFilter = filter),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 3),

                  // Ползунок зума
                  SizedBox(
                    height: 24,
                    child: Row(
                      children: [
                        const Icon(
                          Icons.zoom_in,
                          size: 15,
                          color: Colors.white70,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          '${_currentZoom.toStringAsFixed(1)}x',
                          style: const TextStyle(fontSize: 11),
                        ),
                        Expanded(
                          child: Slider(
                            value: _currentZoom,
                            min: _minZoom,
                            max: _maxZoom,
                            activeColor: Colors.cyanAccent,
                            onChanged: _updateZoom,
                          ),
                        ),
                      ],
                    ),
                  ),

                  // Спинбоксы Яркости и Четкости (шаг 0.1 + ручной ввод) + АВТО
                  Row(
                    children: [
                      Expanded(
                        child: LabSpinBox(
                          key: const ValueKey('spin_brightness'),
                          label: 'Ярк',
                          value: _brightness,
                          min: -10.0,
                          max: 10.0,
                          step: 0.1,
                          onChanged: _updateBrightness,
                        ),
                      ),
                      const SizedBox(width: 6),
                      Expanded(
                        child: LabSpinBox(
                          key: const ValueKey('spin_sharpness'),
                          label: 'Четк',
                          value: _sharpness,
                          min: 0.0,
                          max: 10.0,
                          step: 0.1,
                          onChanged: _updateSharpness,
                        ),
                      ),
                      const SizedBox(width: 6),
                      ElevatedButton(
                        onPressed: _isProcessing ? null : _autoTuneImage,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal.shade700,
                          foregroundColor: Colors.white,
                          minimumSize: const Size(48, 34),
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                        ),
                        child: const Text(
                          'АВТО',
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),

                  // Нижний ряд: Считать капсулы | Мульти-расчет | Съемка/Сохранение | Галерея/Сброс
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      // Быстрый подсчет капсул при текущих параметрах
                      ElevatedButton.icon(
                        onPressed: _isProcessing
                            ? null
                            : () => _runCapsuleDetection(
                                brightnessValues: [_brightness],
                                sharpnessValues: [_sharpness],
                              ),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green.shade700,
                          foregroundColor: Colors.white,
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          minimumSize: const Size(0, 36),
                        ),
                        icon: const Icon(Icons.grain, size: 16),
                        label: const Text(
                          'Считать',
                          style: TextStyle(fontSize: 11.5),
                        ),
                      ),

                      // Мульти-расчет (среднее по нескольким яркостям и четкостям)
                      OutlinedButton.icon(
                        onPressed: _isProcessing ? null : _openMultiPassDialog,
                        style: OutlinedButton.styleFrom(
                          foregroundColor: Colors.cyanAccent,
                          side: const BorderSide(color: Colors.cyanAccent),
                          padding: const EdgeInsets.symmetric(horizontal: 8),
                          minimumSize: const Size(0, 36),
                        ),
                        icon: const Icon(Icons.functions, size: 16),
                        label: const Text(
                          'Мульти',
                          style: TextStyle(fontSize: 11.5),
                        ),
                      ),

                      // Центральная кнопка: Сделать снимок / Сохранить отчет с капсулами в галерею
                      GestureDetector(
                        onTap: _isProcessing ? null : _captureOrSaveImage,
                        child: Container(
                          width: 48,
                          height: 48,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _isProcessing
                                ? Colors.grey
                                : (isPhotoMode
                                      ? Colors.cyanAccent
                                      : Colors.white),
                            border: Border.all(
                              color: Colors.cyanAccent,
                              width: 2.5,
                            ),
                          ),
                          child: _isProcessing
                              ? const Padding(
                                  padding: EdgeInsets.all(12.0),
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2.5,
                                    color: Colors.black,
                                  ),
                                )
                              : Icon(
                                  isPhotoMode
                                      ? Icons.save_alt
                                      : Icons.camera_alt,
                                  color: Colors.black,
                                  size: 24,
                                ),
                        ),
                      ),

                      // Фокус (для камеры) или К камере (для фото)
                      if (!isPhotoMode)
                        OutlinedButton(
                          onPressed: _toggleFocusLock,
                          style: OutlinedButton.styleFrom(
                            foregroundColor: _isFocusLocked
                                ? Colors.amberAccent
                                : Colors.white70,
                            side: BorderSide(
                              color: _isFocusLocked
                                  ? Colors.amberAccent
                                  : Colors.white24,
                            ),
                            padding: const EdgeInsets.symmetric(horizontal: 8),
                            minimumSize: const Size(0, 36),
                          ),
                          child: Text(
                            _isFocusLocked ? 'БЛОК' : 'Фокус',
                            style: const TextStyle(fontSize: 11),
                          ),
                        )
                      else
                        OutlinedButton(
                          onPressed: _returnToLiveCamera,
                          style: OutlinedButton.styleFrom(
                            foregroundColor: Colors.amberAccent,
                            side: const BorderSide(color: Colors.amberAccent),
                            padding: const EdgeInsets.symmetric(horizontal: 8),
                            minimumSize: const Size(0, 36),
                          ),
                          child: const Text(
                            'Камера',
                            style: TextStyle(fontSize: 11),
                          ),
                        ),

                      // Сброс параметров и разметки
                      IconButton(
                        visualDensity: VisualDensity.compact,
                        tooltip: 'Сброс настроек',
                        onPressed: () {
                          FocusManager.instance.primaryFocus?.unfocus();
                          _updateBrightness(0.0);
                          _updateSharpness(1.0);
                          setState(() {
                            _selectedFilter = LabFilter.standard;
                            _capsules = [];
                            _multiPassAverageCount = null;
                          });
                        },
                        icon: const Icon(
                          Icons.refresh,
                          size: 20,
                          color: Colors.white70,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Отрисовщик контуров распознанных капсул и их диаметров в реальном времени
class CapsuleOverlayPainter extends CustomPainter {
  final List<DetectedCapsule> capsules;
  final double scaleMicrons;

  CapsuleOverlayPainter({required this.capsules, required this.scaleMicrons});

  @override
  void paint(Canvas canvas, Size size) {
    final Paint circlePaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final Paint centerPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.fill;

    for (int i = 0; i < capsules.length; i++) {
      final c = capsules[i];
      final double cx = c.cxNorm * size.width;
      final double cy = c.cyNorm * size.height;
      final double r = c.rNorm * size.width;
      final double diamUm = c.diameterMicrons(scaleMicrons);

      canvas.drawCircle(Offset(cx, cy), r, circlePaint);
      canvas.drawCircle(Offset(cx, cy), 2.5, centerPaint);

      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: '${diamUm.toStringAsFixed(0)} мкм',
          style: const TextStyle(
            color: Colors.yellowAccent,
            fontSize: 10,
            fontWeight: FontWeight.bold,
            backgroundColor: Color.fromRGBO(0, 0, 0, 0.6),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      tp.paint(canvas, Offset(cx - tp.width / 2, cy - r - tp.height - 2));
    }
  }

  @override
  bool shouldRepaint(covariant CapsuleOverlayPainter oldDelegate) => true;
}

// Спинбокс с поддержкой ручного ввода и шагом 0.1
class LabSpinBox extends StatefulWidget {
  final String label;
  final double value;
  final double min;
  final double max;
  final double step;
  final ValueChanged<double> onChanged;

  const LabSpinBox({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    this.step = 0.1,
    required this.onChanged,
  });

  @override
  State<LabSpinBox> createState() => _LabSpinBoxState();
}

class _LabSpinBoxState extends State<LabSpinBox> {
  late TextEditingController _textController;
  late FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _textController = TextEditingController(
      text: widget.value.toStringAsFixed(1),
    );
    _focusNode = FocusNode();

    _focusNode.addListener(() {
      if (!_focusNode.hasFocus) {
        _submitManualValue(_textController.text);
      }
    });
  }

  @override
  void didUpdateWidget(covariant LabSpinBox oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!_focusNode.hasFocus &&
        (oldWidget.value - widget.value).abs() > 0.001) {
      _textController.text = widget.value.toStringAsFixed(1);
    }
  }

  @override
  void dispose() {
    _focusNode.dispose();
    _textController.dispose();
    super.dispose();
  }

  void _submitManualValue(String rawText) {
    final normalized = rawText.trim().replaceAll(',', '.');
    final parsed = double.tryParse(normalized);

    if (parsed != null) {
      final clamped = parsed.clamp(widget.min, widget.max);
      final rounded = double.parse(clamped.toStringAsFixed(1));
      _textController.text = rounded.toStringAsFixed(1);
      widget.onChanged(rounded);
    } else {
      _textController.text = widget.value.toStringAsFixed(1);
    }
  }

  void _stepChange(double delta) {
    FocusManager.instance.primaryFocus?.unfocus();
    final nextVal = (widget.value + delta).clamp(widget.min, widget.max);
    final rounded = double.parse(nextVal.toStringAsFixed(1));
    _textController.text = rounded.toStringAsFixed(1);
    widget.onChanged(rounded);
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 34,
      padding: const EdgeInsets.symmetric(horizontal: 6),
      decoration: BoxDecoration(
        color: const Color.fromRGBO(30, 30, 30, 0.95),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.white24),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            '${widget.label}:',
            style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w500),
          ),
          InkWell(
            onTap: () => _stepChange(-widget.step),
            child: Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Icon(Icons.remove, size: 14, color: Colors.white),
            ),
          ),
          Expanded(
            child: TextField(
              controller: _textController,
              focusNode: _focusNode,
              keyboardType: const TextInputType.numberWithOptions(
                signed: true,
                decimal: true,
              ),
              textInputAction: TextInputAction.done,
              textAlign: TextAlign.center,
              style: const TextStyle(
                fontSize: 12.5,
                fontWeight: FontWeight.bold,
                color: Colors.cyanAccent,
              ),
              decoration: const InputDecoration(
                isDense: true,
                contentPadding: EdgeInsets.symmetric(vertical: 4),
                border: InputBorder.none,
              ),
              onTapOutside: (_) => _focusNode.unfocus(),
              onSubmitted: (val) {
                _submitManualValue(val);
                _focusNode.unfocus();
              },
            ),
          ),
          InkWell(
            onTap: () => _stepChange(widget.step),
            child: Container(
              padding: const EdgeInsets.all(4),
              decoration: BoxDecoration(
                color: Colors.white12,
                borderRadius: BorderRadius.circular(4),
              ),
              child: const Icon(Icons.add, size: 14, color: Colors.white),
            ),
          ),
        ],
      ),
    );
  }
}

class MicroscopeGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final Paint paint = Paint()
      ..color = const Color.fromRGBO(0, 255, 255, 0.35)
      ..strokeWidth = 1.1
      ..style = PaintingStyle.stroke;

    final double cx = size.width / 2;
    final double cy = size.height / 2;
    final double radius = math.min(size.width, size.height) * 0.42;

    canvas.drawCircle(Offset(cx, cy), radius, paint);
    canvas.drawLine(Offset(cx - radius, cy), Offset(cx + radius, cy), paint);
    canvas.drawLine(Offset(cx, cy - radius), Offset(cx, cy + radius), paint);

    final double step = radius / 5;
    for (int i = -4; i <= 4; i++) {
      if (i == 0) continue;
      final double d = i * step;
      canvas.drawLine(Offset(cx + d, cy - 4), Offset(cx + d, cy + 4), paint);
      canvas.drawLine(Offset(cx - 4, cy + d), Offset(cx + 4, cy + d), paint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
