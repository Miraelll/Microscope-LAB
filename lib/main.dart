import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:gal/gal.dart';
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await SystemChrome.setPreferredOrientations([DeviceOrientation.portraitUp]);

  final cameras = await availableCameras();
  final camera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.back,
    orElse: () =>
        cameras.isEmpty ? throw Exception('Камеры не найдены') : cameras.first,
  );

  runApp(MicroscopeApp(camera: camera));
}

enum LabFilter {
  standard('Стандарт'),
  highDetail('Контуры HD'),
  coldLight('Чистый белый'),
  achromat('Ахромат Ч/Б'),
  darkField('Темное поле');

  final String title;
  const LabFilter(this.title);
}

class MicroscopeApp extends StatelessWidget {
  final CameraDescription camera;

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
  final CameraDescription camera;

  const CameraScreen({super.key, required this.camera});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController _controller;
  bool _isCameraInitialized = false;

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

  String _substanceName = 'Образец №1';
  String _magnification = '40x';

  // Список доступных увеличений (включая сохраненные пользователем)
  List<String> _magnificationList = ['4x', '10x', '20x', '40x', '100x'];

  @override
  void initState() {
    super.initState();
    _loadCustomMagnifications();
    _initCamera();
  }

  // Загрузка кастомных увеличений из памяти телефона
  Future<void> _loadCustomMagnifications() async {
    final prefs = await SharedPreferences.getInstance();
    final savedList = prefs.getStringList('custom_magnifications');
    if (savedList != null && savedList.isNotEmpty) {
      setState(() {
        _magnificationList = savedList;
      });
    }
  }

  // Сохранение нового увеличения в память
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
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.veryHigh,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await _controller.initialize();
      await _controller.lockCaptureOrientation(DeviceOrientation.portraitUp);
      await _controller.setFlashMode(FlashMode.off);

      try {
        _minExposure = await _controller.getMinExposureOffset();
        _maxExposure = await _controller.getMaxExposureOffset();
        _minZoom = await _controller.getMinZoomLevel();
        _maxZoom = math.min(await _controller.getMaxZoomLevel(), 8.0);
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
    _controller.dispose();
    super.dispose();
  }

  ColorFilter _buildColorFilter({bool isFinalCapture = false}) {
    final double captureBoost = isFinalCapture ? 1.15 : 1.0;
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

    try {
      final double norm = _brightness / 10.0;
      final double expOffset = norm >= 0
          ? norm * _maxExposure * 0.7
          : -norm * _minExposure * 0.7;
      await _controller.setExposureOffset(
        expOffset.clamp(_minExposure, _maxExposure),
      );
    } catch (_) {}
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
    try {
      await _controller.setZoomLevel(clamped);
    } catch (_) {}
  }

  Future<void> _toggleFocusLock() async {
    try {
      final newLockState = !_isFocusLocked;
      await _controller.setFocusMode(
        newLockState ? FocusMode.locked : FocusMode.auto,
      );
      setState(() {
        _isFocusLocked = newLockState;
      });
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          duration: const Duration(seconds: 1),
          content: Text(
            newLockState
                ? 'Фокус ЗАБЛОКИРОВАН (настраивайте резкость микроскопом)'
                : 'Автофокус ВКЛЮЧЕН',
          ),
        ),
      );
    } catch (_) {}
  }

  Future<void> _onTapPreview(
    TapDownDetails details,
    BoxConstraints constraints,
  ) async {
    if (FocusScope.of(context).hasFocus) {
      FocusScope.of(context).unfocus();
    }

    final offset = Offset(
      (details.localPosition.dx / constraints.maxWidth).clamp(0.0, 1.0),
      (details.localPosition.dy / constraints.maxHeight).clamp(0.0, 1.0),
    );

    setState(() {
      _focusPoint = details.localPosition;
    });

    try {
      await _controller.setFocusPoint(offset);
      await _controller.setExposurePoint(offset);
      await _controller.setFocusMode(FocusMode.locked);
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

  Future<void> _autoTuneImage() async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing) return;
    setState(() => _isProcessing = true);

    try {
      await _controller.setExposureOffset(0.0);
      final XFile sampleFile = await _controller.takePicture();
      final Uint8List bytes = await File(sampleFile.path).readAsBytes();

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
            .pow(math.min(width, height) * 0.35, 2)
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
            labelText: 'Название вещества / препарата',
            hintText: 'Например: Кристалл NaCl, Кровь',
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

  // Диалог выбора и сохранения кастомного увеличения
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
              keyboardType: TextInputType.text,
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

  Future<void> _takePicture() async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing || !_isCameraInitialized) return;

    if (_useTimer) {
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
      final XFile rawFile = await _controller.takePicture();
      final Uint8List rawBytes = await File(rawFile.path).readAsBytes();

      final Uint8List processedBytes = await _processAndStampImage(rawBytes);

      final Directory tempDir = await getTemporaryDirectory();
      final String outPath =
          '${tempDir.path}/micro_${DateTime.now().millisecondsSinceEpoch}.png';
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
          content: Text('Снимок «$_substanceName ($_magnification)» сохранен!'),
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

    final Paint imagePaint = Paint()
      ..colorFilter = _buildColorFilter(isFinalCapture: true)
      ..filterQuality = FilterQuality.high;
    canvas.drawImage(srcImage, Offset.zero, imagePaint);

    final DateTime now = DateTime.now();
    final String dateStr =
        '${now.day.toString().padLeft(2, '0')}.${now.month.toString().padLeft(2, '0')}.${now.year} '
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}';

    final String line1Title = '$_substanceName  |  Увеличение: $_magnification';
    final String line2Params =
        'Фильтр: ${_selectedFilter.title}  |  Ярк: ${_brightness.toStringAsFixed(1)}  |  Четк: ${_sharpness.toStringAsFixed(1)}';
    final String line3Date = 'Дата съемки: $dateStr';

    final double textMaxWidth = w * 0.70;
    final double titleFontSize = w * 0.036;
    final double subFontSize = w * 0.025;

    final TextPainter titlePainter = TextPainter(
      text: TextSpan(
        text: line1Title,
        style: TextStyle(
          color: Colors.white,
          fontSize: titleFontSize,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: textMaxWidth);

    final TextPainter paramsPainter = TextPainter(
      text: TextSpan(
        text: line2Params,
        style: TextStyle(color: Colors.white70, fontSize: subFontSize),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: textMaxWidth);

    final TextPainter datePainter = TextPainter(
      text: TextSpan(
        text: line3Date,
        style: TextStyle(color: Colors.white60, fontSize: subFontSize * 0.95),
      ),
      textDirection: TextDirection.ltr,
    )..layout(maxWidth: textMaxWidth);

    final double verticalPadding = h * 0.018;
    final double lineSpacing = h * 0.006;
    final double totalTextHeight =
        titlePainter.height +
        paramsPainter.height +
        datePainter.height +
        (lineSpacing * 2) +
        (verticalPadding * 2.4);

    final double barHeight = math.max(h * 0.105, totalTextHeight);
    final double barTop = h - barHeight;

    final Rect barRect = Rect.fromLTWH(0, barTop, w.toDouble(), barHeight);
    canvas.drawRect(
      barRect,
      Paint()..color = const Color.fromRGBO(0, 0, 0, 0.80),
    );

    double currentY = barTop + verticalPadding;
    titlePainter.paint(canvas, Offset(w * 0.03, currentY));

    currentY += titlePainter.height + lineSpacing;
    paramsPainter.paint(canvas, Offset(w * 0.03, currentY));

    currentY += paramsPainter.height + lineSpacing;
    datePainter.paint(canvas, Offset(w * 0.03, currentY));

    String scaleLabel = '50 мкм';
    final int cleanMag =
        int.tryParse(_magnification.replaceAll(RegExp(r'[^0-9]'), '')) ?? 40;
    if (cleanMag <= 5) {
      scaleLabel = '500 мкм';
    } else if (cleanMag <= 15) {
      scaleLabel = '200 мкм';
    } else if (cleanMag <= 30) {
      scaleLabel = '100 мкм';
    } else if (cleanMag <= 60) {
      scaleLabel = '50 мкм';
    } else {
      scaleLabel = '20 мкм';
    }

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
      Offset(scaleLeft, scaleY - barHeight * 0.10),
      Offset(scaleLeft, scaleY + barHeight * 0.10),
      scalePaint,
    );
    canvas.drawLine(
      Offset(scaleRight, scaleY - barHeight * 0.10),
      Offset(scaleRight, scaleY + barHeight * 0.10),
      scalePaint,
    );

    final TextPainter scaleTextPainter = TextPainter(
      text: TextSpan(
        text: scaleLabel,
        style: TextStyle(
          color: Colors.cyanAccent,
          fontSize: titleFontSize * 0.85,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    scaleTextPainter.paint(
      canvas,
      Offset(
        scaleLeft + (scaleBarWidth - scaleTextPainter.width) / 2,
        scaleY - scaleTextPainter.height - (barHeight * 0.12),
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
    if (!_isCameraInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Column(
          children: [
            // 1. ВЕРХНЯЯ ПАНЕЛЬ
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
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
                            const SizedBox(width: 5),
                            Expanded(
                              child: Text(
                                _substanceName,
                                overflow: TextOverflow.ellipsis,
                                style: const TextStyle(fontSize: 13),
                              ),
                            ),
                            const Icon(
                              Icons.edit,
                              size: 13,
                              color: Colors.white54,
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  // Кнопка выпадающего списка с кастомным вводом и сохранением
                  InkWell(
                    onTap: _editMagnificationDialog,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 6,
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
                              fontSize: 13,
                            ),
                          ),
                          const SizedBox(width: 4),
                          const Icon(
                            Icons.arrow_drop_down,
                            color: Colors.amberAccent,
                            size: 18,
                          ),
                        ],
                      ),
                    ),
                  ),
                  TextButton(
                    style: TextButton.styleFrom(
                      minimumSize: const Size(40, 32),
                      padding: const EdgeInsets.symmetric(horizontal: 6),
                    ),
                    onPressed: () => setState(() => _isWide9x16 = !_isWide9x16),
                    child: Text(
                      _isWide9x16 ? '9:16' : '3:4',
                      style: const TextStyle(
                        color: Colors.cyanAccent,
                        fontWeight: FontWeight.bold,
                        fontSize: 12,
                      ),
                    ),
                  ),
                  IconButton(
                    visualDensity: VisualDensity.compact,
                    icon: Icon(
                      Icons.grid_on,
                      size: 20,
                      color: _showGrid ? Colors.cyanAccent : Colors.white38,
                    ),
                    onPressed: () => setState(() => _showGrid = !_showGrid),
                  ),
                  IconButton(
                    visualDensity: VisualDensity.compact,
                    icon: Icon(
                      Icons.timer,
                      size: 20,
                      color: _useTimer ? Colors.orangeAccent : Colors.white38,
                    ),
                    onPressed: () => setState(() => _useTimer = !_useTimer),
                  ),
                ],
              ),
            ),

            // 2. ОБЛАСТЬ КАМЕРЫ
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

                    final double previewAspectRatio =
                        1.0 / _controller.value.aspectRatio;

                    return SizedBox(
                      width: viewWidth,
                      height: viewHeight,
                      child: ClipRect(
                        child: GestureDetector(
                          onTapDown: (details) => _onTapPreview(
                            details,
                            BoxConstraints(
                              maxWidth: viewWidth,
                              maxHeight: viewHeight,
                            ),
                          ),
                          child: Stack(
                            alignment: Alignment.center,
                            children: [
                              SizedBox.expand(
                                child: FittedBox(
                                  fit: BoxFit.cover,
                                  child: SizedBox(
                                    width: viewWidth,
                                    height: viewWidth / previewAspectRatio,
                                    child: ColorFiltered(
                                      colorFilter: _buildColorFilter(),
                                      child: CameraPreview(_controller),
                                    ),
                                  ),
                                ),
                              ),

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

            // 3. НИЖНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              color: Colors.black87,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  SizedBox(
                    height: 30,
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
                  const SizedBox(height: 4),

                  SizedBox(
                    height: 26,
                    child: Row(
                      children: [
                        const Icon(
                          Icons.zoom_in,
                          size: 16,
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
                          minimumSize: const Size(52, 36),
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

                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      OutlinedButton.icon(
                        onPressed: _toggleFocusLock,
                        style: OutlinedButton.styleFrom(
                          foregroundColor: _isFocusLocked
                              ? Colors.amberAccent
                              : Colors.white,
                          side: BorderSide(
                            color: _isFocusLocked
                                ? Colors.amberAccent
                                : Colors.white38,
                          ),
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          minimumSize: const Size(0, 36),
                        ),
                        icon: Icon(
                          _isFocusLocked
                              ? Icons.lock
                              : Icons.center_focus_strong,
                          size: 16,
                        ),
                        label: Text(
                          _isFocusLocked ? 'Фокус: БЛОК' : 'Фокус: АВТО',
                          style: const TextStyle(fontSize: 12),
                        ),
                      ),

                      GestureDetector(
                        onTap: _isProcessing ? null : _takePicture,
                        child: Container(
                          width: 52,
                          height: 52,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: _isProcessing ? Colors.grey : Colors.white,
                            border: Border.all(
                              color: Colors.cyanAccent,
                              width: 3,
                            ),
                          ),
                          child: _isProcessing
                              ? const Padding(
                                  padding: EdgeInsets.all(12.0),
                                  child: CircularProgressIndicator(
                                    strokeWidth: 3,
                                    color: Colors.black,
                                  ),
                                )
                              : const Icon(
                                  Icons.camera_alt,
                                  color: Colors.black,
                                  size: 26,
                                ),
                        ),
                      ),

                      TextButton.icon(
                        onPressed: () {
                          FocusManager.instance.primaryFocus?.unfocus();
                          _updateBrightness(0.0);
                          _updateSharpness(1.0);
                          setState(() => _selectedFilter = LabFilter.standard);
                        },
                        icon: const Icon(
                          Icons.refresh,
                          size: 16,
                          color: Colors.white70,
                        ),
                        label: const Text(
                          'Сброс',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
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
      height: 36,
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
                fontSize: 13,
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
      ..color = const Color.fromRGBO(0, 255, 255, 0.45)
      ..strokeWidth = 1.2
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
