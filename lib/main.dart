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
import 'package:tflite_flutter/tflite_flutter.dart';

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

enum LabFilter {
  standard('Стандарт'),
  highDetail('Контуры HD'),
  coldLight('Чистый белый'),
  achromat('Ахромат Ч/Б'),
  darkField('Темное поле');

  final String title;
  const LabFilter(this.title);
}

class DetectedCapsule {
  final double cxNorm;
  final double cyNorm;
  final double rNorm;
  final double score;
  final bool isManual;

  const DetectedCapsule({
    required this.cxNorm,
    required this.cyNorm,
    required this.rNorm,
    this.score = 1.0,
    this.isManual = false,
  });

  DetectedCapsule copyWith({
    double? cxNorm,
    double? cyNorm,
    double? rNorm,
    double? score,
    bool? isManual,
  }) {
    return DetectedCapsule(
      cxNorm: cxNorm ?? this.cxNorm,
      cyNorm: cyNorm ?? this.cyNorm,
      rNorm: rNorm ?? this.rNorm,
      score: score ?? this.score,
      isManual: isManual ?? this.isManual,
    );
  }

  double diameterPx(double imageWidth) => 2.0 * rNorm * imageWidth;

  double diameterMicrons(double scaleMicrons) =>
      2.0 * rNorm * (scaleMicrons / 0.20);
}

class LearnedCapsuleProfile {
  final double targetWallSignal;
  final double targetHollowDiff;
  final double maxAllowedSpike;
  final int minRadiusPx;
  final int maxRadiusPx;
  final int trainingCount;

  const LearnedCapsuleProfile({
    required this.targetWallSignal,
    required this.targetHollowDiff,
    required this.maxAllowedSpike,
    required this.minRadiusPx,
    required this.maxRadiusPx,
    this.trainingCount = 1,
  });
}

class MultiPassTask {
  final Uint8List rgbaBytes;
  final int width;
  final int height;
  final List<double> brightnessList;
  final List<double> sharpnessList;
  final int filterIndex;
  final int minRadiusPx;
  final int maxRadiusPx;
  final double sensitivity;
  final double roiCenterXNorm;
  final double roiCenterYNorm;
  final double roiRadiusNorm;
  final LearnedCapsuleProfile? learnedProfile;
  final List<Offset> bannedZonesNorm;

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
    required this.roiCenterXNorm,
    required this.roiCenterYNorm,
    required this.roiRadiusNorm,
    this.learnedProfile,
    this.bannedZonesNorm = const [],
  });
}

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
// КОНВЕРТАЦИЯ FLOAT16 <-> DOUBLE (ДЛЯ СОВМЕСТИМОСТИ С ЛЮБЫМ ЭКСПОРТОМ TFLITE)
// ============================================================================
int _doubleToFloat16Bits(double value) {
  final bd = ByteData(4)..setFloat32(0, value, Endian.little);
  final int fbits = bd.getUint32(0, Endian.little);
  final int sign = (fbits >> 16) & 0x8000;
  int val = (fbits & 0x7fffffff) + 0x1000;

  if (val >= 0x47800000) {
    if ((fbits & 0x7fffffff) >= 0x47800000) {
      if (val < 0x7f800000) return sign | 0x7c00;
      return sign | 0x7c00 | ((fbits & 0x007fffff) >> 13);
    }
    return sign | 0x7bff;
  }
  if (val >= 0x38800000) {
    return sign | ((val - 0x38000000) >> 13);
  }
  if (val < 0x33000000) return sign;
  val = (fbits & 0x7fffffff) >> 23;
  return sign |
      ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >> (val - 102))) >>
          (126 - val));
}

double _float16BitsToDouble(int hbits) {
  int mant = hbits & 0x03ff;
  int exp = hbits & 0x7c00;
  if (exp == 0x7c00) {
    exp = 0x3fc00;
  } else if (exp != 0) {
    exp += 0x1c000;
  } else if (mant != 0) {
    exp = 0x1c400;
    do {
      mant <<= 1;
      exp -= 0x400;
    } while ((mant & 0x400) == 0);
    mant &= 0x3ff;
  }
  final int fbits = ((hbits & 0x8000) << 16) | ((exp | mant) << 13);
  final bd = ByteData(4)..setUint32(0, fbits, Endian.little);
  return bd.getFloat32(0, Endian.little);
}

// ============================================================================
// АЛГОРИТМ ПОДСЧЕТА КАПСУЛ (ПО ПРИНЦИПУ CAPSULECOUNTER: DoG + МОРФОЛОГИЯ)
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
        roiCenterXNorm: task.roiCenterXNorm,
        roiCenterYNorm: task.roiCenterYNorm,
        roiRadiusNorm: task.roiRadiusNorm,
        learnedProfile: task.learnedProfile,
        bannedZonesNorm: task.bannedZonesNorm,
      );
      passes.add(
        SinglePassSummary(brightness: b, sharpness: s, capsules: capsules),
      );
    }
  }

  if (passes.isEmpty) {
    return const MultiPassResult(
      bestCapsules: [],
      averageCount: 0.0,
      minCount: 0,
      maxCount: 0,
      bestBrightness: 0.0,
      bestSharpness: 1.0,
      passes: [],
    );
  }

  double total = 0.0;
  int minC = passes.first.capsules.length;
  int maxC = passes.first.capsules.length;

  for (final p in passes) {
    final int c = p.capsules.length;
    total += c.toDouble();
    if (c < minC) minC = c;
    if (c > maxC) maxC = c;
  }

  final double avgCount = total / passes.length.toDouble();

  SinglePassSummary bestPass = passes.first;
  double minDiff = (bestPass.capsules.length.toDouble() - avgCount).abs();

  for (final p in passes) {
    final double diff = (p.capsules.length.toDouble() - avgCount).abs();
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

Float32List _computeBoxBlur(Float32List src, int w, int h, int radius) {
  final Float32List temp = Float32List(w * h);
  final Float32List out = Float32List(w * h);
  final double invDiam = 1.0 / (2 * radius + 1).toDouble();

  for (int y = 0; y < h; y++) {
    final int rowOffset = y * w;
    double sum = src[rowOffset] * (radius + 1).toDouble();
    for (int x = 1; x <= radius; x++) {
      sum += src[rowOffset + x.clamp(0, w - 1)];
    }
    for (int x = 0; x < w; x++) {
      temp[rowOffset + x] = sum * invDiam;
      final int outIdx = (x - radius).clamp(0, w - 1);
      final int inIdx = (x + radius + 1).clamp(0, w - 1);
      sum += src[rowOffset + inIdx] - src[rowOffset + outIdx];
    }
  }

  for (int x = 0; x < w; x++) {
    double sum = temp[x] * (radius + 1).toDouble();
    for (int y = 1; y <= radius; y++) {
      sum += temp[y.clamp(0, h - 1) * w + x];
    }
    for (int y = 0; y < h; y++) {
      out[y * w + x] = sum * invDiam;
      final int outIdx = (y - radius).clamp(0, h - 1);
      final int inIdx = (y + radius + 1).clamp(0, h - 1);
      sum += temp[inIdx * w + x] - temp[outIdx * w + x];
    }
  }

  return out;
}

// Морфологический фильтр связных компонент: находит длинные волокна/палочки
// по моментам инерции и вытянутости (Eccentricity / Major Axis) и блокирует их
Uint8List _buildFiberAndDebrisMask(
  Float32List darkSignal,
  Uint8List safeMask,
  int w,
  int h,
  double fiberThresh,
  int maxCapsuleRadius,
) {
  final int numPixels = w * h;
  final Uint8List visited = Uint8List(numPixels);
  final Uint8List badMask = Uint8List(numPixels);
  final Int32List queue = Int32List(numPixels);

  for (int i = 0; i < numPixels; i++) {
    if (safeMask[i] == 0 || visited[i] == 1) continue;
    if (darkSignal[i] < fiberThresh) continue;

    int head = 0;
    int tail = 0;
    queue[tail++] = i;
    visited[i] = 1;

    int minX = i % w;
    int maxX = minX;
    int minY = i ~/ w;
    int maxY = minY;
    double sumX = 0.0;
    double sumY = 0.0;
    double sumWeight = 0.0;

    while (head < tail) {
      final int curr = queue[head++];
      final int cx = curr % w;
      final int cy = curr ~/ w;
      final double weight = darkSignal[curr];

      if (cx < minX) minX = cx;
      if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy;
      if (cy > maxY) maxY = cy;

      sumX += cx.toDouble() * weight;
      sumY += cy.toDouble() * weight;
      sumWeight += weight;

      for (int dy = -1; dy <= 1; dy++) {
        final int ny = cy + dy;
        if (ny <= 1 || ny >= h - 1) continue;
        for (int dx = -1; dx <= 1; dx++) {
          final int nx = cx + dx;
          if (nx <= 1 || nx >= w - 1) continue;
          final int nIdx = ny * w + nx;
          if (visited[nIdx] == 0 &&
              safeMask[nIdx] == 1 &&
              darkSignal[nIdx] >= fiberThresh * 0.75) {
            visited[nIdx] = 1;
            queue[tail++] = nIdx;
          }
        }
      }
    }

    final int count = tail;
    if (count < 4) continue;

    final double meanX = sumX / sumWeight;
    final double meanY = sumY / sumWeight;
    double mu20 = 0.0;
    double mu02 = 0.0;
    double mu11 = 0.0;
    double maxSigInComp = 0.0;

    for (int k = 0; k < count; k++) {
      final int idx = queue[k];
      final double dx = (idx % w).toDouble() - meanX;
      final double dy = (idx ~/ w).toDouble() - meanY;
      final double wt = darkSignal[idx];
      if (wt > maxSigInComp) maxSigInComp = wt;
      mu20 += wt * dx * dx;
      mu02 += wt * dy * dy;
      mu11 += wt * dx * dy;
    }

    mu20 /= sumWeight;
    mu02 /= sumWeight;
    mu11 /= sumWeight;

    final double diff = mu20 - mu02;
    final double root = math.sqrt(diff * diff + 4.0 * mu11 * mu11);
    final double lambda1 = math.max(0.1, (mu20 + mu02 + root) * 0.5);
    final double lambda2 = math.max(0.1, (mu20 + mu02 - root) * 0.5);
    final double elongation = math.sqrt(lambda1 / lambda2);

    final int spanW = maxX - minX + 1;
    final int spanH = maxY - minY + 1;
    final double diag = math.sqrt((spanW * spanW + spanH * spanH).toDouble());

    // Критерий ворсинки / палочки / черного пятна грязи:
    // 1) Вытянутая темная структура (elongation > 2.35 и диагональ > 16px)
    // 2) Слишком резкое черное пятно пыли (maxSigInComp > fiberThresh * 2.2 и малая площадь)
    final bool isFiberStick =
        (elongation > 2.35 && diag > 15.0) ||
        (diag > maxCapsuleRadius.toDouble() * 3.2 && elongation > 1.95) ||
        (maxSigInComp > fiberThresh * 2.4 && count < 45 && elongation > 1.5);

    if (isFiberStick) {
      const int halo = 5;
      for (int k = 0; k < count; k++) {
        final int idx = queue[k];
        final int px = idx % w;
        final int py = idx ~/ w;
        for (int hy = -halo; hy <= halo; hy++) {
          final int yy = py + hy;
          if (yy < 0 || yy >= h) continue;
          for (int hx = -halo; hx <= halo; hx++) {
            final int xx = px + hx;
            if (xx < 0 || xx >= w) continue;
            badMask[yy * w + xx] = 1;
          }
        }
      }
    }
  }

  return badMask;
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
  required double roiCenterXNorm,
  required double roiCenterYNorm,
  required double roiRadiusNorm,
  LearnedCapsuleProfile? learnedProfile,
  List<Offset> bannedZonesNorm = const [],
}) {
  final int numPixels = w * h;
  final Float32List rawLum = Float32List(numPixels);
  final Uint8List validMask = Uint8List(numPixels);

  final double cx = roiCenterXNorm * w.toDouble();
  final double cy = roiCenterYNorm * h.toDouble();
  final double minSide = math.min(w, h).toDouble();
  final double effectiveRadius = minSide * roiRadiusNorm * 0.90;
  final double maxSearchRadiusSq = effectiveRadius * effectiveRadius;

  // 1. Выделение светлого круга окуляра (отсечение черной рамки и синей хроматической каймы)
  for (int y = 0; y < h; y++) {
    final double dy = y.toDouble() - cy;
    for (int x = 0; x < w; x++) {
      final double dx = x.toDouble() - cx;
      final int p = y * w + x;
      final int idx = p * 4;

      final double r = rgba[idx].toDouble();
      final double g = rgba[idx + 1].toDouble();
      final double b = rgba[idx + 2].toDouble();

      final double lum = 0.299 * r + 0.587 * g + 0.114 * b;
      rawLum[p] = lum;

      if (dx * dx + dy * dy <= maxSearchRadiusSq &&
          lum > 140.0 &&
          (b - r) < 25.0) {
        validMask[p] = 1;
      }
    }
  }

  // Безопасный отступ от края линзы
  final Uint8List safeMask = Uint8List(numPixels);
  const int borderMargin = 10;
  for (int y = borderMargin; y < h - borderMargin; y++) {
    for (int x = borderMargin; x < w - borderMargin; x++) {
      final int p = y * w + x;
      if (validMask[p] == 0) continue;
      if (validMask[p - borderMargin] == 1 &&
          validMask[p + borderMargin] == 1 &&
          validMask[p - borderMargin * w] == 1 &&
          validMask[p + borderMargin * w] == 1 &&
          validMask[p - 7 * w - 7] == 1 &&
          validMask[p - 7 * w + 7] == 1 &&
          validMask[p + 7 * w - 7] == 1 &&
          validMask[p + 7 * w + 7] == 1) {
        safeMask[p] = 1;
      }
    }
  }

  // Блокируем точки, которые пользователь удалил вручную в режиме правки
  for (final bz in bannedZonesNorm) {
    final int bx = (bz.dx * w.toDouble()).round();
    final int by = (bz.dy * h.toDouble()).round();
    const int banRad = 11;
    for (int dy = -banRad; dy <= banRad; dy++) {
      final int yy = by + dy;
      if (yy < 0 || yy >= h) continue;
      for (int dx = -banRad; dx <= banRad; dx++) {
        final int xx = bx + dx;
        if (xx < 0 || xx >= w) continue;
        if (dx * dx + dy * dy <= banRad * banRad) {
          safeMask[yy * w + xx] = 0;
        }
      }
    }
  }

  // 2. Разность гауссианов (DoG): подавляет высокочастотный шум и неравномерную засветку лампы
  final Float32List smoothSmall = _computeBoxBlur(rawLum, w, h, 1); // 3x3
  final Float32List localBg = _computeBoxBlur(smoothSmall, w, h, 16); // 33x33

  final Float32List darkSignal = Float32List(numPixels);
  final Float32List gradMag = Float32List(numPixels);

  final double paramBoost = (1.0 + (sharpness - 1.0) * 0.10 - brightness * 0.04)
      .clamp(0.75, 1.60);

  double sumSig = 0.0;
  double sumSigSq = 0.0;
  int validCount = 0;

  for (int p = 0; p < numPixels; p++) {
    if (safeMask[p] == 0) continue;
    final double diff = (localBg[p] - smoothSmall[p]) * paramBoost;
    final double sig = diff > 0.0 ? diff : 0.0;
    darkSignal[p] = sig;

    if (sig < 15.0) {
      sumSig += sig;
      sumSigSq += sig * sig;
      validCount++;
    }
  }

  if (validCount < 300) return [];

  for (int y = 2; y < h - 2; y++) {
    for (int x = 2; x < w - 2; x++) {
      final int p = y * w + x;
      if (safeMask[p] == 0) continue;
      final double gx = smoothSmall[p + 1] - smoothSmall[p - 1];
      final double gy = smoothSmall[p + w] - smoothSmall[p - w];
      gradMag[p] = math.sqrt(gx * gx + gy * gy) * paramBoost;
    }
  }

  final double meanSig = sumSig / validCount.toDouble();
  final double stdSig = math.sqrt(
    math.max(0.0, (sumSigSq / validCount.toDouble()) - meanSig * meanSig),
  );

  // Адаптивный порог стенки для полупрозрачных капсул (как в CapsuleCounter)
  double wallThreshold = math.max(
    1.15 - sensitivity * 0.45,
    meanSig + stdSig * (1.35 - sensitivity * 0.65),
  );

  int effMinR = minR.clamp(4, 26);
  int effMaxR = maxR.clamp(effMinR + 1, 34);

  if (learnedProfile != null) {
    wallThreshold = math.max(
      0.95,
      learnedProfile.targetWallSignal * (0.65 - (sensitivity - 0.5) * 0.25),
    );
    effMinR = learnedProfile.minRadiusPx.clamp(4, 28);
    effMaxR = learnedProfile.maxRadiusPx.clamp(effMinR + 1, 36);
  }

  // 3. Морфологическое выделение и блокировка длинных палочек / волокон / темной грязи
  final double fiberThresh = math.max(5.5, wallThreshold * 2.6);
  final Uint8List fiberMask = _buildFiberAndDebrisMask(
    darkSignal,
    safeMask,
    w,
    h,
    fiberThresh,
    effMaxR,
  );

  const int numAngles = 12;
  final List<double> cosTable = List.generate(
    numAngles,
    (i) => math.cos(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
  );
  final List<double> sinTable = List.generate(
    numAngles,
    (i) => math.sin(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
  );

  final List<DetectedCapsule> rawCandidates = [];
  final List<double> rayPeakSig = List<double>.filled(numAngles, 0.0);
  final List<int> rayBestR = List<int>.filled(numAngles, 0);
  final List<double> rayExtSig = List<double>.filled(numAngles, 0.0);

  // 4. Поиск кольцевых и овальных капсул (включая слипшиеся в гроздьях и восьмерках)
  for (int cyCurr = effMaxR + 6; cyCurr < h - effMaxR - 6; cyCurr++) {
    for (int cxCurr = effMaxR + 6; cxCurr < w - effMaxR - 6; cxCurr++) {
      final int centerIdx = cyCurr * w + cxCurr;
      if (safeMask[centerIdx] == 0 || fiberMask[centerIdx] == 1) continue;

      final double centerSig =
          (darkSignal[centerIdx] * 2.0 +
              darkSignal[centerIdx - 1] +
              darkSignal[centerIdx + 1] +
              darkSignal[centerIdx - w] +
              darkSignal[centerIdx + w]) /
          6.0;

      // Центр прозрачной капсулы не должен быть слишком темным пятном
      if (centerSig > fiberThresh * 0.85) continue;

      double bestRScore = -1.0;
      double bestRNorm = effMinR.toDouble() / w.toDouble();

      for (int r = effMinR; r <= effMaxR; r++) {
        final double rD = r.toDouble();
        // Эластичный пояс ±25% позволяет находить овальные и деформированные капсулы
        final int rStart = math.max(3, (rD * 0.78).round());
        final int rEnd = math.min(effMaxR + 3, (rD * 1.22).round());

        bool hitsFiber = false;
        int activeWallRays = 0;
        int hollowRays = 0;
        double sumGrad = 0.0;
        double sumRadius = 0.0;

        for (int a = 0; a < numAngles; a++) {
          final double ca = cosTable[a];
          final double sa = sinTable[a];

          double maxS = -1.0;
          double maxG = 0.0;
          int bestRR = r;

          for (int rr = rStart; rr <= rEnd; rr++) {
            final int px = (cxCurr.toDouble() + rr.toDouble() * ca).round();
            final int py = (cyCurr.toDouble() + rr.toDouble() * sa).round();
            final int pIdx = py * w + px;

            if (fiberMask[pIdx] == 1) {
              hitsFiber = true;
              break;
            }

            final double s = darkSignal[pIdx] + gradMag[pIdx] * 0.30;
            if (s > maxS) {
              maxS = s;
              maxG = gradMag[pIdx];
              bestRR = rr;
            }
          }

          if (hitsFiber) break;

          rayPeakSig[a] = maxS;
          rayBestR[a] = bestRR;
          sumGrad += maxG;
          sumRadius += bestRR.toDouble();

          if (maxS >= wallThreshold * 0.78) {
            activeWallRays++;
          }
          if (maxS > centerSig + wallThreshold * 0.28) {
            hollowRays++;
          }

          final double rOut = bestRR.toDouble() + 3.0;
          final int ox = (cxCurr.toDouble() + rOut * ca).round().clamp(
            0,
            w - 1,
          );
          final int oy = (cyCurr.toDouble() + rOut * sa).round().clamp(
            0,
            h - 1,
          );
          rayExtSig[a] = darkSignal[oy * w + ox];
        }

        if (hitsFiber) continue;

        // Для одиночных и слипшихся в гроздья капсул достаточно 7 активных лучей из 12
        final int minRequiredRays = (8.0 - sensitivity * 2.5).round().clamp(
          6,
          8,
        );
        if (activeWallRays < minRequiredRays || hollowRays < minRequiredRays) {
          continue;
        }

        // Проверяем, что хотя бы в 3 из 4 квадрантов (или во всех 4 со смягченным порогом) есть стенка
        final double qTop =
            (rayPeakSig[0] + rayPeakSig[1] + rayPeakSig[11]) / 3.0;
        final double qRight =
            (rayPeakSig[2] + rayPeakSig[3] + rayPeakSig[4]) / 3.0;
        final double qBottom =
            (rayPeakSig[5] + rayPeakSig[6] + rayPeakSig[7]) / 3.0;
        final double qLeft =
            (rayPeakSig[8] + rayPeakSig[9] + rayPeakSig[10]) / 3.0;

        int activeQuadrants = 0;
        if (qTop >= wallThreshold * 0.65) activeQuadrants++;
        if (qRight >= wallThreshold * 0.65) activeQuadrants++;
        if (qBottom >= wallThreshold * 0.65) activeQuadrants++;
        if (qLeft >= wallThreshold * 0.65) activeQuadrants++;
        if (activeQuadrants < 3) continue;

        final List<double> sortedRing = List<double>.from(rayPeakSig)..sort();
        final double medianRing = (sortedRing[5] + sortedRing[6]) * 0.5;
        if (medianRing < wallThreshold * 0.80) continue;

        // Внешний фон: берем 7 самых светлых внешних направлений из 12 (чтобы видеть капсулы в гроздьях!)
        final List<double> sortedExt = List<double>.from(rayExtSig)..sort();
        double freeExtSum = 0.0;
        for (int k = 0; k < 7; k++) {
          freeExtSum += sortedExt[k];
        }
        final double meanFreeExt = freeExtSum / 7.0;

        if (meanFreeExt > medianRing * 0.78) continue;

        final double hollowDiff = math.max(0.0, medianRing - centerSig);
        final double extDiff = math.max(0.0, medianRing - meanFreeExt);

        final double score =
            (hollowDiff * 4.5) +
            (extDiff * 4.0) +
            (sumGrad / numAngles.toDouble()) * 0.6 +
            (activeWallRays.toDouble() * 1.6) +
            (activeQuadrants == 4 ? 4.0 : 0.0);

        if (score > bestRScore) {
          bestRScore = score;
          bestRNorm = (sumRadius / numAngles.toDouble()) / w.toDouble();
        }
      }

      final double minAcceptScore = (11.5 - sensitivity * 3.5).clamp(7.5, 13.0);
      if (bestRScore >= minAcceptScore) {
        rawCandidates.add(
          DetectedCapsule(
            cxNorm: cxCurr.toDouble() / w.toDouble(),
            cyNorm: cyCurr.toDouble() / h.toDouble(),
            rNorm: bestRNorm,
            score: bestRScore,
          ),
        );
      }
    }
  }

  // 5. Подавление перекрытий (NMS): сохраняет соприкасающиеся капсулы в цепочках и восьмерках
  rawCandidates.sort((a, b) => b.score.compareTo(a.score));
  final List<DetectedCapsule> accepted = [];
  final double aspect = h.toDouble() / w.toDouble();

  for (final cand in rawCandidates) {
    bool overlaps = false;
    for (final ex in accepted) {
      final double dx = cand.cxNorm - ex.cxNorm;
      final double dy = (cand.cyNorm - ex.cyNorm) * aspect;
      final double dist = math.sqrt(dx * dx + dy * dy);
      final double minAllowedDist = (cand.rNorm + ex.rNorm) * 0.68;
      if (dist < minAllowedDist) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      accepted.add(cand);
      if (accepted.length >= 250) break;
    }
  }

  return accepted;
}

// ============================================================================
// ОТОБРАЖЕНИЕ И ИНТЕРФЕЙС ПРИЛОЖЕНИЯ
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

  Interpreter? _tfliteInterpreter;
  bool _isTfliteLoaded = false;
  bool _useAiModel = true;

  Uint8List? _loadedImageBytes;
  int _loadedImageWidth = 1080;
  int _loadedImageHeight = 1920;

  static const int _workResolutionWidth = 640;
  Uint8List? _cachedWorkRgba;
  int _cachedWorkW = 0;
  int _cachedWorkH = 0;
  Float32List? _snapDarkSignal;
  bool _snapBufferDirty = true;

  double _brightness = 0.0;
  double _sharpness = 1.0;

  LabFilter _selectedFilter = LabFilter.standard;
  bool _isWide9x16 = true;

  double _minExposure = -2.0;
  double _maxExposure = 2.0;
  double _minZoom = 1.0;
  double _maxZoom = 20.0;
  double _currentZoom = 1.0;
  Offset _photoPanOffset = Offset.zero;

  double _baseZoomOnScaleStart = 1.0;
  double _baseGridRadiusOnScaleStart = 0.36;

  double _gridCenterXNorm = 0.5;
  double _gridCenterYNorm = 0.5;
  double _gridRadiusNorm = 0.36;
  bool _adjustGridMode = false;

  bool _isFocusLocked = false;
  Offset? _focusPoint;

  bool _showGrid = true;
  bool _useTimer = false;
  int _countdown = 0;
  bool _isProcessing = false;

  String _substanceName = 'Капсулы №1';
  String _magnification = '40x';
  List<String> _magnificationList = ['4x', '10x', '20x', '40x', '100x'];

  List<DetectedCapsule> _capsules = [];
  final List<Offset> _bannedNoiseZonesNorm = [];
  double? _multiPassAverageCount;
  List<SinglePassSummary> _lastMultiPassDetails = [];
  bool _showCapsuleOverlay = true;
  bool _manualEditCapsules = false;
  bool _magneticSnapEnabled = true;

  int? _selectedCapsuleIndex;
  int? _draggingCapsuleIndex;
  double _manualBrushRNorm = 0.010;

  LearnedCapsuleProfile? _learnedProfile;

  String _multiBrightnessInput = '-0.5 0.0 0.5';
  String _multiSharpnessInput = '1.0 1.5 2.0';
  int _minCapsuleRadiusPx = 4;
  int _maxCapsuleRadiusPx = 14;
  double _detectionSensitivity = 0.60;

  @override
  void initState() {
    super.initState();
    _loadSavedSettings();
    _initTfliteModel();
    if (widget.camera != null) {
      _initCamera();
    }
  }

  // Безопасная инициализация TFLite с обязательным вызовом allocateTensors()
  Future<void> _initTfliteModel() async {
    try {
      final options = InterpreterOptions()..threads = 4;
      Interpreter? interpreter;
      try {
        interpreter = await Interpreter.fromAsset(
          'assets/capsule_detector.tflite',
          options: options,
        );
      } catch (_) {
        interpreter = await Interpreter.fromAsset(
          'capsule_detector.tflite',
          options: options,
        );
      }

      interpreter.allocateTensors();

      if (mounted) {
        setState(() {
          _tfliteInterpreter = interpreter;
          _isTfliteLoaded = true;
          _useAiModel = true;
        });
      }
      debugPrint(
        'YOLOv8 TFLite готова! Вход: ${interpreter.getInputTensor(0).shape} (${interpreter.getInputTensor(0).type}), '
        'Выход: ${interpreter.getOutputTensor(0).shape} (${interpreter.getOutputTensor(0).type})',
      );
    } catch (e) {
      debugPrint('Модель TFLite не загружена (используется алгоритм): $e');
      if (mounted) {
        setState(() {
          _isTfliteLoaded = false;
          _useAiModel = false;
        });
      }
    }
  }

  // ==========================================================================
  // БЕЗОПАСНЫЙ ЗАПУСК НЕЙРОСЕТИ YOLOv8 ЧЕРЕЗ ПРЯМОЙ БУФЕР ТЕНЗОРОВ
  // (Исключает ошибку "Bad state: failed precondition")
  // ==========================================================================
  List<DetectedCapsule> _runYoloTfliteSinglePass({
    required Uint8List srcRgba,
    required int srcW,
    required int srcH,
    required double brightness,
    required double sharpness,
    required int filterIndex,
    required double sensitivity,
  }) {
    final interpreter = _tfliteInterpreter;
    if (interpreter == null) return [];

    // Гарантируем, что тензоры аллоцированы перед записью
    interpreter.allocateTensors();

    final Tensor inTensor = interpreter.getInputTensor(0);
    final Tensor outTensor = interpreter.getOutputTensor(0);

    final List<int> inShape = inTensor.shape;
    // Определяем формат входа: NCHW [1, 3, 640, 640] или NHWC [1, 640, 640, 3]
    final bool isInputNchw = inShape.length == 4 && inShape[1] == 3;
    final int modelH = isInputNchw
        ? inShape[2]
        : (inShape.length >= 3 ? inShape[1] : 640);
    final int modelW = isInputNchw
        ? inShape[3]
        : (inShape.length >= 3 ? inShape[2] : 640);

    final int totalInElements = modelW * modelH * 3;
    final Float32List rgbNormalized = Float32List(totalInElements);

    double contrast = 1.0 + (sharpness * 0.14);
    double brightnessOffset = brightness * 8.0;
    double rScale = 1.0;
    double gScale = 1.0;
    double bScale = 1.0;

    if (filterIndex == LabFilter.highDetail.index) {
      contrast *= 1.28;
      brightnessOffset -= 10.0;
    } else if (filterIndex == LabFilter.coldLight.index) {
      rScale = 0.92;
      gScale = 0.98;
      bScale = 1.14;
      contrast *= 1.08;
    } else if (filterIndex == LabFilter.achromat.index) {
      contrast *= 1.15;
    }
    final bool isInverted = filterIndex == LabFilter.darkField.index;
    final bool isGray = filterIndex == LabFilter.achromat.index;

    final int planeSize = modelW * modelH;

    // Прямое масштабирование из рабочей матрицы srcRgba в размер входа нейросети
    for (int y = 0; y < modelH; y++) {
      final int sy = ((y * srcH) ~/ modelH).clamp(0, srcH - 1);
      for (int x = 0; x < modelW; x++) {
        final int sx = ((x * srcW) ~/ modelW).clamp(0, srcW - 1);
        final int srcIdx = (sy * srcW + sx) * 4;

        double r = srcRgba[srcIdx].toDouble();
        double g = srcRgba[srcIdx + 1].toDouble();
        double b = srcRgba[srcIdx + 2].toDouble();

        if (isInverted) {
          r = 255.0 - r;
          g = 255.0 - g;
          b = 255.0 - b;
        }

        r = ((r - 128.0) * (contrast * rScale) + 128.0 + brightnessOffset)
            .clamp(0.0, 255.0);
        g = ((g - 128.0) * (contrast * gScale) + 128.0 + brightnessOffset)
            .clamp(0.0, 255.0);
        b = ((b - 128.0) * (contrast * bScale) + 128.0 + brightnessOffset)
            .clamp(0.0, 255.0);

        if (isGray) {
          final double lum = (0.299 * r + 0.587 * g + 0.114 * b).clamp(
            0.0,
            255.0,
          );
          r = lum;
          g = lum;
          b = lum;
        }

        final double rn = r / 255.0;
        final double gn = g / 255.0;
        final double bn = b / 255.0;

        final int pixelIndex = y * modelW + x;
        if (isInputNchw) {
          rgbNormalized[pixelIndex] = rn;
          rgbNormalized[planeSize + pixelIndex] = gn;
          rgbNormalized[2 * planeSize + pixelIndex] = bn;
        } else {
          final int base = pixelIndex * 3;
          rgbNormalized[base] = rn;
          rgbNormalized[base + 1] = gn;
          rgbNormalized[base + 2] = bn;
        }
      }
    }

    // Упаковываем входной буфер точно под размер байтов входного тензора (Float32 / Float16 / Uint8)
    final int inBytesLen = inTensor.numBytes();
    final Uint8List rawInputBytes = Uint8List(inBytesLen);

    if (inBytesLen == totalInElements * 4) {
      // Float32
      rawInputBytes.buffer.asUint8List().setAll(
        0,
        rgbNormalized.buffer.asUint8List(),
      );
    } else if (inBytesLen == totalInElements * 2) {
      // Float16 (если экспортировано с half=True)
      final ByteData bd = ByteData.sublistView(rawInputBytes);
      for (int i = 0; i < totalInElements; i++) {
        bd.setUint16(
          i * 2,
          _doubleToFloat16Bits(rgbNormalized[i]),
          Endian.little,
        );
      }
    } else if (inBytesLen == totalInElements) {
      // Uint8
      for (int i = 0; i < totalInElements; i++) {
        rawInputBytes[i] = (rgbNormalized[i] * 255.0).round().clamp(0, 255);
      }
    } else {
      final Uint8List fBytes = rgbNormalized.buffer.asUint8List();
      final int copyLen = math.min(inBytesLen, fBytes.length);
      rawInputBytes.setRange(0, copyLen, fBytes);
    }

    // Копируем байты напрямую в нативный тензор и вызываем invoke()
    inTensor.data = rawInputBytes;
    interpreter.invoke();
    final Uint8List rawOutBytes = Uint8List.fromList(outTensor.data);

    // Декодируем выходной тензор (поддерживаем и Float32, и Float16)
    final List<int> outShape = outTensor.shape;
    int totalOutElements = 1;
    for (final dim in outShape) {
      totalOutElements *= dim;
    }

    final Float32List outputFloats = Float32List(totalOutElements);
    if (rawOutBytes.lengthInBytes >= totalOutElements * 4) {
      final ByteData bd = ByteData.sublistView(rawOutBytes);
      for (int i = 0; i < totalOutElements; i++) {
        outputFloats[i] = bd.getFloat32(i * 4, Endian.little);
      }
    } else if (rawOutBytes.lengthInBytes >= totalOutElements * 2) {
      final ByteData bd = ByteData.sublistView(rawOutBytes);
      for (int i = 0; i < totalOutElements; i++) {
        outputFloats[i] = _float16BitsToDouble(
          bd.getUint16(i * 2, Endian.little),
        );
      }
    }

    final int dim1 = outShape.length >= 3 ? outShape[1] : 5;
    final int dim2 = outShape.length >= 3 ? outShape[2] : 8400;
    final bool isChannelsFirst = dim1 < dim2; // [1, 5, 8400] vs [1, 8400, 5]
    final int numAnchors = isChannelsFirst ? dim2 : dim1;
    final int numChannels = isChannelsFirst ? dim1 : dim2;

    // Для моделей, обученных на малых выборках, порог уверенности делается адаптивным
    final double confThreshold = (0.32 - sensitivity * 0.26).clamp(0.06, 0.50);

    final double aspect = srcH.toDouble() / math.max(1.0, srcW.toDouble());
    final double roiCx = _gridCenterXNorm;
    final double roiCy = _gridCenterYNorm;
    final double roiMaxR = _gridRadiusNorm * 0.92;

    final List<DetectedCapsule> rawDetections = [];

    for (int i = 0; i < numAnchors; i++) {
      double maxClassScore = 0.0;
      for (int c = 4; c < numChannels; c++) {
        final double score = isChannelsFirst
            ? outputFloats[c * numAnchors + i]
            : outputFloats[i * numChannels + c];
        if (score > maxClassScore) {
          maxClassScore = score;
        }
      }

      if (maxClassScore < confThreshold) continue;

      double cx = isChannelsFirst
          ? outputFloats[0 * numAnchors + i]
          : outputFloats[i * numChannels + 0];
      double cy = isChannelsFirst
          ? outputFloats[1 * numAnchors + i]
          : outputFloats[i * numChannels + 1];
      double w = isChannelsFirst
          ? outputFloats[2 * numAnchors + i]
          : outputFloats[i * numChannels + 2];
      double h = isChannelsFirst
          ? outputFloats[3 * numAnchors + i]
          : outputFloats[i * numChannels + 3];

      if (cx > 1.5 || cy > 1.5 || w > 1.5 || h > 1.5) {
        cx /= modelW.toDouble();
        cy /= modelH.toDouble();
        w /= modelW.toDouble();
        h /= modelH.toDouble();
      }

      if (cx <= 0.02 || cx >= 0.98 || cy <= 0.02 || cy >= 0.98) continue;

      final double dxRoi = cx - roiCx;
      final double dyRoi = (cy - roiCy) * aspect;
      if (math.sqrt(dxRoi * dxRoi + dyRoi * dyRoi) > roiMaxR) continue;

      bool inBannedZone = false;
      for (final bz in _bannedNoiseZonesNorm) {
        if ((bz.dx - cx).abs() < 0.018 && (bz.dy - cy).abs() < 0.018) {
          inBannedZone = true;
          break;
        }
      }
      if (inBannedZone) continue;

      final double rNorm = ((w + h) / 4.0).clamp(0.004, 0.065);

      rawDetections.add(
        DetectedCapsule(
          cxNorm: cx,
          cyNorm: cy,
          rNorm: rNorm,
          score: maxClassScore * 100.0,
        ),
      );
    }

    rawDetections.sort((a, b) => b.score.compareTo(a.score));
    final List<DetectedCapsule> accepted = [];

    for (final cand in rawDetections) {
      bool overlaps = false;
      for (final ex in accepted) {
        final double dx = cand.cxNorm - ex.cxNorm;
        final double dy = (cand.cyNorm - ex.cyNorm) * aspect;
        final double dist = math.sqrt(dx * dx + dy * dy);
        final double minAllowedDist = (cand.rNorm + ex.rNorm) * 0.68;
        if (dist < minAllowedDist) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) {
        accepted.add(cand);
        if (accepted.length >= 250) break;
      }
    }

    return accepted;
  }

  Future<void> _loadSavedSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final savedList = prefs.getStringList('custom_magnifications');
    if (savedList != null && savedList.isNotEmpty) {
      setState(() {
        _magnificationList = savedList;
      });
    }

    final double? wallSig = prefs.getDouble('learned_wall_sig');
    final double? hollowDiff = prefs.getDouble('learned_hollow_diff');
    final double? maxSpike = prefs.getDouble('learned_max_spike');
    final int? minR = prefs.getInt('learned_min_r');
    final int? maxR = prefs.getInt('learned_max_r');
    final int? trainCnt = prefs.getInt('learned_train_count');

    if (wallSig != null && hollowDiff != null && minR != null && maxR != null) {
      setState(() {
        _learnedProfile = LearnedCapsuleProfile(
          targetWallSignal: wallSig,
          targetHollowDiff: hollowDiff,
          maxAllowedSpike: maxSpike ?? 14.0,
          minRadiusPx: minR,
          maxRadiusPx: maxR,
          trainingCount: trainCnt ?? 1,
        );
        _minCapsuleRadiusPx = minR;
        _maxCapsuleRadiusPx = maxR;
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
        _maxZoom = 20.0;
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
    _tfliteInterpreter?.close();
    _controller?.dispose();
    super.dispose();
  }

  double _getScaleMicrons() {
    final int cleanMag =
        int.tryParse(_magnification.replaceAll(RegExp(r'[^0-9]'), '')) ?? 40;
    if (cleanMag <= 5) return 500.0;
    if (cleanMag <= 15) return 200.0;
    if (cleanMag <= 30) return 100.0;
    if (cleanMag <= 60) return 50.0;
    return 20.0;
  }

  Map<String, double> _getDiameterStatsMicrons() {
    if (_capsules.isEmpty) {
      return {'mean': 0.0, 'min': 0.0, 'max': 0.0};
    }
    final double scaleUm = _getScaleMicrons();
    double sum = 0.0;
    double minD = double.infinity;
    double maxD = 0.0;

    for (final c in _capsules) {
      final double d = c.diameterMicrons(scaleUm);
      sum += d;
      if (d < minD) minD = d;
      if (d > maxD) maxD = d;
    }

    return {
      'mean': sum / _capsules.length.toDouble(),
      'min': minD,
      'max': maxD,
    };
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
    final double clamped = newValue.clamp(-10.0, 10.0);
    final double rounded = double.parse(clamped.toStringAsFixed(1));
    if (_brightness != rounded) {
      setState(() {
        _brightness = rounded;
        _snapBufferDirty = true;
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
    final double clamped = newValue.clamp(0.0, 10.0);
    final double rounded = double.parse(clamped.toStringAsFixed(1));
    if (_sharpness != rounded) {
      setState(() {
        _sharpness = rounded;
        _snapBufferDirty = true;
      });
    }
  }

  Future<void> _updateZoom(double newValue) async {
    final double clamped = newValue.clamp(_minZoom, _maxZoom);
    setState(() {
      _currentZoom = clamped;
      if (_currentZoom <= 1.02) {
        _photoPanOffset = Offset.zero;
      }
    });
    if (_loadedImageBytes == null && _controller != null) {
      try {
        final double hwMax = await _controller!.getMaxZoomLevel();
        await _controller!.setZoomLevel(clamped.clamp(_minZoom, hwMax));
      } catch (_) {}
    }
  }

  void _updateManualCircleRadiusNorm(double newRNorm) {
    final double clampedR = newRNorm.clamp(0.003, 0.055);
    setState(() {
      _manualBrushRNorm = clampedR;
      if (_selectedCapsuleIndex != null &&
          _selectedCapsuleIndex! >= 0 &&
          _selectedCapsuleIndex! < _capsules.length) {
        final int idx = _selectedCapsuleIndex!;
        _capsules[idx] = _capsules[idx].copyWith(
          rNorm: clampedR,
          isManual: true,
        );
      }
    });
  }

  void _ensureSnapBufferReady() {
    if (!_snapBufferDirty && _snapDarkSignal != null) return;
    final Uint8List? rgba = _cachedWorkRgba;
    final int w = _cachedWorkW;
    final int h = _cachedWorkH;
    if (rgba == null || w <= 0 || h <= 0) return;

    final int numPixels = w * h;
    final Float32List rawLum = Float32List(numPixels);

    for (int i = 0; i < numPixels; i++) {
      final int idx = i * 4;
      final double r = rgba[idx].toDouble();
      final double g = rgba[idx + 1].toDouble();
      final double b = rgba[idx + 2].toDouble();
      rawLum[i] = 0.299 * r + 0.587 * g + 0.114 * b;
    }

    final Float32List localBg = _computeBoxBlur(rawLum, w, h, 16);
    final Float32List darkSig = Float32List(numPixels);
    for (int i = 0; i < numPixels; i++) {
      final double d = localBg[i] - rawLum[i];
      darkSig[i] = d > 0.0 ? d : 0.0;
    }

    _snapDarkSignal = darkSig;
    _snapBufferDirty = false;
  }

  DetectedCapsule _createCapsuleWithMagneticSnap(
    double tapXNorm,
    double tapYNorm, {
    int? ignoreIndex,
    double? preserveRNorm,
  }) {
    final double targetRNorm = preserveRNorm ?? _manualBrushRNorm;

    if (!_magneticSnapEnabled || _cachedWorkRgba == null) {
      return DetectedCapsule(
        cxNorm: tapXNorm,
        cyNorm: tapYNorm,
        rNorm: targetRNorm,
        isManual: true,
      );
    }

    _ensureSnapBufferReady();
    final Float32List? darkSig = _snapDarkSignal;
    final int w = _cachedWorkW;
    final int h = _cachedWorkH;
    if (darkSig == null || w <= 0 || h <= 0) {
      return DetectedCapsule(
        cxNorm: tapXNorm,
        cyNorm: tapYNorm,
        rNorm: targetRNorm,
        isManual: true,
      );
    }

    final int tapX = (tapXNorm * w.toDouble()).round().clamp(10, w - 11);
    final int tapY = (tapYNorm * h.toDouble()).round().clamp(10, h - 11);

    const int snapSearchRadius = 12;
    final int baseR = (targetRNorm * w.toDouble()).round().clamp(3, 30);
    final int minR = (baseR - 3).clamp(3, 28);
    final int maxR = (baseR + 3).clamp(minR + 1, 32);

    const int numAngles = 12;
    final List<double> cosT = List.generate(
      numAngles,
      (i) => math.cos(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
    );
    final List<double> sinT = List.generate(
      numAngles,
      (i) => math.sin(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
    );

    double bestScore = -1.0;
    int bestCx = tapX;
    int bestCy = tapY;
    int bestR = baseR;

    final double aspect = h.toDouble() / w.toDouble();

    for (int dy = -snapSearchRadius; dy <= snapSearchRadius; dy++) {
      final int cy = tapY + dy;
      if (cy <= maxR + 2 || cy >= h - maxR - 2) continue;

      for (int dx = -snapSearchRadius; dx <= snapSearchRadius; dx++) {
        final int distSq = dx * dx + dy * dy;
        if (distSq > snapSearchRadius * snapSearchRadius) continue;

        final int cx = tapX + dx;
        if (cx <= maxR + 2 || cx >= w - maxR - 2) continue;

        final double candXNorm = cx.toDouble() / w.toDouble();
        final double candYNorm = cy.toDouble() / h.toDouble();

        bool tooCloseToExisting = false;
        for (int i = 0; i < _capsules.length; i++) {
          if (i == ignoreIndex) continue;
          final ex = _capsules[i];
          final double ddx = candXNorm - ex.cxNorm;
          final double ddy = (candYNorm - ex.cyNorm) * aspect;
          if (math.sqrt(ddx * ddx + ddy * ddy) < ex.rNorm * 0.72) {
            tooCloseToExisting = true;
            break;
          }
        }
        if (tooCloseToExisting) continue;

        final double proximityWeight =
            1.0 -
            0.35 * (math.sqrt(distSq.toDouble()) / snapSearchRadius.toDouble());
        final double centerSig = darkSig[cy * w + cx];

        for (int r = minR; r <= maxR; r++) {
          final double rD = r.toDouble();
          double ringSum = 0.0;
          int activeAngles = 0;

          for (int a = 0; a < numAngles; a++) {
            final double ca = cosT[a];
            final double sa = sinT[a];
            final int p1 =
                (cy.toDouble() + rD * sa).round() * w +
                (cx.toDouble() + rD * ca).round();
            final double sig = darkSig[p1];
            ringSum += sig;
            if (sig > centerSig + 0.5) activeAngles++;
          }

          if (activeAngles < 6) continue;

          final double meanRing = ringSum / numAngles.toDouble();
          final double hollowContrast = meanRing - centerSig;
          if (hollowContrast <= 0.4) continue;

          final double candidateScore =
              (hollowContrast * 3.5 + activeAngles.toDouble() * 2.0) *
              proximityWeight;

          if (candidateScore > bestScore) {
            bestScore = candidateScore;
            bestCx = cx;
            bestCy = cy;
            bestR = r;
          }
        }
      }
    }

    if (bestScore > 9.0) {
      final double snappedRNorm =
          preserveRNorm ?? (bestR.toDouble() / w.toDouble());
      _manualBrushRNorm = snappedRNorm;
      return DetectedCapsule(
        cxNorm: bestCx.toDouble() / w.toDouble(),
        cyNorm: bestCy.toDouble() / h.toDouble(),
        rNorm: snappedRNorm,
        score: bestScore,
        isManual: true,
      );
    }

    return DetectedCapsule(
      cxNorm: tapXNorm,
      cyNorm: tapYNorm,
      rNorm: targetRNorm,
      isManual: true,
    );
  }

  Future<void> _trainFromMarkedCapsules() async {
    if (_capsules.isEmpty || _cachedWorkRgba == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text(
            'Сначала отметьте пальцем хотя бы 1–3 реальные капсулы на снимке!',
          ),
        ),
      );
      return;
    }

    _ensureSnapBufferReady();
    final Float32List? darkSig = _snapDarkSignal;
    final int w = _cachedWorkW;
    final int h = _cachedWorkH;
    if (darkSig == null || w <= 0 || h <= 0) return;

    const int numAngles = 12;
    final List<double> cosT = List.generate(
      numAngles,
      (i) => math.cos(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
    );
    final List<double> sinT = List.generate(
      numAngles,
      (i) => math.sin(i.toDouble() * 2.0 * math.pi / numAngles.toDouble()),
    );

    double sumWallSig = 0.0;
    double sumHollowDiff = 0.0;
    double maxObservedCapsulePeak = 0.0;
    int minR = 999;
    int maxR = 0;
    int sampleCount = 0;

    for (final c in _capsules) {
      final int cx = (c.cxNorm * w.toDouble()).round().clamp(10, w - 11);
      final int cy = (c.cyNorm * h.toDouble()).round().clamp(10, h - 11);
      final int r = (c.rNorm * w.toDouble()).round().clamp(3, 30);

      final double centerSig = darkSig[cy * w + cx];
      final int rStart = math.max(3, (r.toDouble() * 0.75).round());
      final int rEnd = math.min(34, (r.toDouble() * 1.25).round());

      double ringPeakSum = 0.0;
      for (int a = 0; a < numAngles; a++) {
        final double ca = cosT[a];
        final double sa = sinT[a];
        double rayMax = 0.0;
        for (int rr = rStart; rr <= rEnd; rr++) {
          final int px = (cx.toDouble() + rr.toDouble() * ca).round().clamp(
            0,
            w - 1,
          );
          final int py = (cy.toDouble() + rr.toDouble() * sa).round().clamp(
            0,
            h - 1,
          );
          final double s = darkSig[py * w + px];
          if (s > rayMax) rayMax = s;
        }
        ringPeakSum += rayMax;
        if (rayMax > maxObservedCapsulePeak) {
          maxObservedCapsulePeak = rayMax;
        }
      }

      final double meanRingPeak = ringPeakSum / numAngles.toDouble();
      sumWallSig += meanRingPeak;
      sumHollowDiff += math.max(0.5, meanRingPeak - centerSig);
      if (r < minR) minR = r;
      if (r > maxR) maxR = r;
      sampleCount++;
    }

    if (sampleCount == 0) return;

    double learnedWall = (sumWallSig / sampleCount.toDouble()).clamp(1.1, 14.0);
    double learnedHollow = (sumHollowDiff / sampleCount.toDouble()).clamp(
      0.5,
      12.0,
    );
    double learnedMaxSpike = math
        .max(8.0, maxObservedCapsulePeak * 1.55)
        .clamp(8.0, 22.0);

    final int learnedMinR = (minR - 1).clamp(3, 26);
    final int learnedMaxR = (maxR + 2).clamp(learnedMinR + 1, 34);
    final int nextTrainCount = (_learnedProfile?.trainingCount ?? 0) + 1;

    final profile = LearnedCapsuleProfile(
      targetWallSignal: learnedWall,
      targetHollowDiff: learnedHollow,
      maxAllowedSpike: learnedMaxSpike,
      minRadiusPx: learnedMinR,
      maxRadiusPx: learnedMaxR,
      trainingCount: nextTrainCount,
    );

    final prefs = await SharedPreferences.getInstance();
    await prefs.setDouble('learned_wall_sig', profile.targetWallSignal);
    await prefs.setDouble('learned_hollow_diff', profile.targetHollowDiff);
    await prefs.setDouble('learned_max_spike', profile.maxAllowedSpike);
    await prefs.setInt('learned_min_r', profile.minRadiusPx);
    await prefs.setInt('learned_max_r', profile.maxRadiusPx);
    await prefs.setInt('learned_train_count', profile.trainingCount);

    final List<DetectedCapsule> userVerifiedCapsules = _capsules
        .map((c) => c.copyWith(isManual: true))
        .toList();

    setState(() {
      _learnedProfile = profile;
      _minCapsuleRadiusPx = learnedMinR;
      _maxCapsuleRadiusPx = learnedMaxR;
    });

    await _runCapsuleDetection(
      brightnessValues: [_brightness],
      sharpnessValues: [_sharpness],
      preservedManualCapsules: userVerifiedCapsules,
    );

    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          backgroundColor: Colors.deepPurple.shade700,
          content: Text(
            'Обучение #$nextTrainCount применено! Эталонов: $sampleCount, всего найдено: ${_capsules.length} шт.',
          ),
        ),
      );
    }
  }

  Future<void> _autoFitGridToOcular(Uint8List bytes) async {
    try {
      final ui.Codec codec = await ui.instantiateImageCodec(
        bytes,
        targetWidth: 160,
        targetHeight: 160,
      );
      final ui.FrameInfo frame = await codec.getNextFrame();
      final ByteData? rgba = await frame.image.toByteData(
        format: ui.ImageByteFormat.rawRgba,
      );
      if (rgba == null) return;

      final Uint8List px = rgba.buffer.asUint8List();
      final int w = frame.image.width;
      final int h = frame.image.height;

      double sumX = 0.0;
      double sumY = 0.0;
      int brightCount = 0;

      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final int idx = (y * w + x) * 4;
          final double lum =
              0.299 * px[idx] + 0.587 * px[idx + 1] + 0.114 * px[idx + 2];
          if (lum > 120.0) {
            sumX += x.toDouble();
            sumY += y.toDouble();
            brightCount++;
          }
        }
      }

      if (brightCount > (w * h * 0.05)) {
        final double cxNorm = (sumX / brightCount.toDouble()) / w.toDouble();
        final double cyNorm = (sumY / brightCount.toDouble()) / h.toDouble();
        final double areaNorm = brightCount.toDouble() / (w * h).toDouble();
        final double estimatedRadius = math.sqrt(areaNorm / math.pi) * 1.12;

        if (mounted) {
          setState(() {
            _gridCenterXNorm = cxNorm.clamp(0.2, 0.8);
            _gridCenterYNorm = cyNorm.clamp(0.2, 0.8);
            _gridRadiusNorm = estimatedRadius.clamp(0.18, 0.46);
          });
        }
      }
    } catch (_) {}
  }

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
    final ui.Codec fullCodec = await ui.instantiateImageCodec(bytes);
    final ui.FrameInfo fullFrame = await fullCodec.getNextFrame();

    final int workWidth = _workResolutionWidth;
    final int workHeight =
        ((workWidth * fullFrame.image.height) / fullFrame.image.width)
            .round()
            .clamp(320, 1180);

    final ui.Codec workCodec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: workWidth,
      targetHeight: workHeight,
    );
    final ui.FrameInfo workFrame = await workCodec.getNextFrame();
    final ByteData? workRgba = await workFrame.image.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );

    setState(() {
      _loadedImageBytes = bytes;
      _loadedImageWidth = fullFrame.image.width;
      _loadedImageHeight = fullFrame.image.height;
      _cachedWorkRgba = workRgba != null
          ? Uint8List.fromList(workRgba.buffer.asUint8List())
          : null;
      _cachedWorkW = workFrame.image.width;
      _cachedWorkH = workFrame.image.height;
      _snapBufferDirty = true;
      _capsules = [];
      _bannedNoiseZonesNorm.clear();
      _selectedCapsuleIndex = null;
      _multiPassAverageCount = null;
      _lastMultiPassDetails = [];
      _currentZoom = 1.0;
      _photoPanOffset = Offset.zero;
      _showGrid = true;
    });

    await _autoFitGridToOcular(bytes);
  }

  void _returnToLiveCamera() {
    FocusManager.instance.primaryFocus?.unfocus();
    setState(() {
      _loadedImageBytes = null;
      _cachedWorkRgba = null;
      _snapDarkSignal = null;
      _capsules = [];
      _bannedNoiseZonesNorm.clear();
      _selectedCapsuleIndex = null;
      _multiPassAverageCount = null;
      _lastMultiPassDetails = [];
      _manualEditCapsules = false;
      _adjustGridMode = false;
      _photoPanOffset = Offset.zero;
    });
  }

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
  // ЕДИНЫЙ ЗАПУСК РАСПОЗНАВАНИЯ: ГИБРИДНЫЙ ИИ YOLOv8 + CAPSULECOUNTER АЛГОРИТМ
  // ==========================================================================
  Future<void> _runCapsuleDetection({
    required List<double> brightnessValues,
    required List<double> sharpnessValues,
    List<DetectedCapsule>? preservedManualCapsules,
  }) async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_isProcessing) return;
    setState(() {
      _isProcessing = true;
      _adjustGridMode = false;
    });

    try {
      final Uint8List? sourceBytes = await _obtainCurrentImageBytes();
      if (sourceBytes == null || _cachedWorkRgba == null) return;

      final task = MultiPassTask(
        rgbaBytes: _cachedWorkRgba!,
        width: _cachedWorkW,
        height: _cachedWorkH,
        brightnessList: brightnessValues,
        sharpnessList: sharpnessValues,
        filterIndex: _selectedFilter.index,
        minRadiusPx: _minCapsuleRadiusPx,
        maxRadiusPx: _maxCapsuleRadiusPx,
        sensitivity: _detectionSensitivity,
        roiCenterXNorm: _gridCenterXNorm,
        roiCenterYNorm: _gridCenterYNorm,
        roiRadiusNorm: _gridRadiusNorm,
        learnedProfile: _learnedProfile,
        bannedZonesNorm: List<Offset>.from(_bannedNoiseZonesNorm),
      );

      // Запускаем контурно-морфологический детектор CapsuleCounter в Isolate
      final MultiPassResult algoResult = await compute(
        _runMultiPassIsolate,
        task,
      );

      MultiPassResult finalResult = algoResult;
      final double aspect =
          _cachedWorkH.toDouble() / math.max(1.0, _cachedWorkW.toDouble());

      // Если включен режим ИИ (YOLOv8 TFLite), запускаем нейросеть и объединяем с контурами
      if (_isTfliteLoaded && _useAiModel && _tfliteInterpreter != null) {
        try {
          final List<SinglePassSummary> aiPasses = [];

          for (int pIdx = 0; pIdx < algoResult.passes.length; pIdx++) {
            final p = algoResult.passes[pIdx];
            final List<DetectedCapsule> yoloCaps = _runYoloTfliteSinglePass(
              srcRgba: _cachedWorkRgba!,
              srcW: _cachedWorkW,
              srcH: _cachedWorkH,
              brightness: p.brightness,
              sharpness: p.sharpness,
              filterIndex: _selectedFilter.index,
              sensitivity: _detectionSensitivity,
            );

            // Гибридное слияние: берем боксы нейросети YOLOv8 и дополняем контурами CapsuleCounter
            final List<DetectedCapsule> combined = List<DetectedCapsule>.from(
              yoloCaps,
            );
            for (final ac in p.capsules) {
              bool exists = false;
              for (final yc in combined) {
                final double dx = ac.cxNorm - yc.cxNorm;
                final double dy = (ac.cyNorm - yc.cyNorm) * aspect;
                if (math.sqrt(dx * dx + dy * dy) <
                    (ac.rNorm + yc.rNorm) * 0.72) {
                  exists = true;
                  break;
                }
              }
              if (!exists) {
                combined.add(ac);
              }
            }

            aiPasses.add(
              SinglePassSummary(
                brightness: p.brightness,
                sharpness: p.sharpness,
                capsules: combined,
              ),
            );
          }

          if (aiPasses.isNotEmpty) {
            double total = 0.0;
            int minC = aiPasses.first.capsules.length;
            int maxC = aiPasses.first.capsules.length;
            for (final p in aiPasses) {
              final int c = p.capsules.length;
              total += c.toDouble();
              if (c < minC) minC = c;
              if (c > maxC) maxC = c;
            }
            final double avgCount = total / aiPasses.length.toDouble();
            SinglePassSummary bestPass = aiPasses.first;
            double minDiff = (bestPass.capsules.length.toDouble() - avgCount)
                .abs();
            for (final p in aiPasses) {
              final double diff = (p.capsules.length.toDouble() - avgCount)
                  .abs();
              if (diff < minDiff) {
                minDiff = diff;
                bestPass = p;
              }
            }

            finalResult = MultiPassResult(
              bestCapsules: bestPass.capsules,
              averageCount: avgCount,
              minCount: minC,
              maxCount: maxC,
              bestBrightness: bestPass.brightness,
              bestSharpness: bestPass.sharpness,
              passes: aiPasses,
            );
          }
        } catch (aiErr) {
          debugPrint(
            'Предупреждение TFLite (использован контурный расчет): $aiErr',
          );
        }
      }

      if (!mounted) return;

      final List<DetectedCapsule> mergedCapsules = [];
      if (preservedManualCapsules != null &&
          preservedManualCapsules.isNotEmpty) {
        mergedCapsules.addAll(preservedManualCapsules);
      }

      for (final autoCap in finalResult.bestCapsules) {
        bool overlapsWithManual = false;
        for (final manCap in mergedCapsules) {
          final double dx = autoCap.cxNorm - manCap.cxNorm;
          final double dy = (autoCap.cyNorm - manCap.cyNorm) * aspect;
          if (math.sqrt(dx * dx + dy * dy) <
              (autoCap.rNorm + manCap.rNorm) * 0.72) {
            overlapsWithManual = true;
            break;
          }
        }
        if (!overlapsWithManual) {
          mergedCapsules.add(autoCap);
        }
      }

      setState(() {
        _capsules = mergedCapsules;
        _selectedCapsuleIndex = _capsules.isNotEmpty
            ? _capsules.length - 1
            : null;
        _multiPassAverageCount = finalResult.averageCount;
        _lastMultiPassDetails = finalResult.passes;
        _showCapsuleOverlay = true;
        _brightness = finalResult.bestBrightness;
        _sharpness = finalResult.bestSharpness;
        _snapBufferDirty = true;
      });

      final stats = _getDiameterStatsMicrons();
      final bool isMulti = finalResult.passes.length > 1;
      final String modeLabel = (_isTfliteLoaded && _useAiModel)
          ? 'ИИ YOLOv8'
          : 'Алгоритм';

      if (preservedManualCapsules == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            duration: const Duration(seconds: 3),
            backgroundColor: Colors.teal.shade800,
            content: Text(
              isMulti
                  ? '[$modeLabel] Мульти (${finalResult.passes.length} прох.): Ср = ${finalResult.averageCount.toStringAsFixed(1)} шт. '
                        '(выбрано: ${_capsules.length} шт.) | Ø ср: ${stats['mean']!.toStringAsFixed(1)} мкм'
                  : '[$modeLabel] Найдено капсул: ${_capsules.length} шт. | Ø ср: ${stats['mean']!.toStringAsFixed(1)} мкм',
            ),
          ),
        );
      }
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
      final double? v = double.tryParse(p);
      if (v != null) {
        res.add(double.parse(v.clamp(minVal, maxVal).toStringAsFixed(1)));
      }
    }
    return res.isEmpty ? [fallback] : res;
  }

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
                    'Укажите несколько значений через пробел:',
                    style: TextStyle(fontSize: 12, color: Colors.white70),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: bCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Значения Яркости (через пробел)',
                      hintText: '-0.5 0.0 0.5',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                  const SizedBox(height: 10),
                  TextField(
                    controller: sCtrl,
                    decoration: const InputDecoration(
                      labelText: 'Значения Четкости (через пробел)',
                      hintText: '1.0 1.5 2.0',
                      border: OutlineInputBorder(),
                      isDense: true,
                    ),
                  ),
                  const Divider(),
                  Text(
                    'Радиус капсул (при 640px): $tempMinR – $tempMaxR пикс.',
                    style: const TextStyle(fontSize: 12),
                  ),
                  RangeSlider(
                    values: RangeValues(
                      tempMinR.toDouble(),
                      tempMaxR.toDouble(),
                    ),
                    min: 3,
                    max: 32,
                    divisions: 29,
                    activeColor: Colors.cyanAccent,
                    labels: RangeLabels('$tempMinR px', '$tempMaxR px'),
                    onChanged: (vals) {
                      setDialogState(() {
                        tempMinR = vals.start.round();
                        tempMaxR = math.max(tempMinR + 1, vals.end.round());
                      });
                    },
                  ),
                  Text(
                    'Чувствительность (ИИ / контуры): ${(tempSens * 100.0).round()}%',
                    style: const TextStyle(fontSize: 12),
                  ),
                  Slider(
                    value: tempSens,
                    min: 0.15,
                    max: 0.90,
                    activeColor: Colors.amberAccent,
                    onChanged: (val) => setDialogState(() => tempSens = val),
                  ),
                  if (_learnedProfile != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 4),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.psychology,
                            size: 16,
                            color: Colors.purpleAccent,
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              'Обучений: ${_learnedProfile!.trainingCount} (R: ${_learnedProfile!.minRadiusPx}–${_learnedProfile!.maxRadiusPx}px)',
                              style: const TextStyle(
                                fontSize: 11,
                                color: Colors.purpleAccent,
                              ),
                            ),
                          ),
                          TextButton(
                            onPressed: () async {
                              final prefs =
                                  await SharedPreferences.getInstance();
                              await prefs.remove('learned_wall_sig');
                              await prefs.remove('learned_train_count');
                              setState(() {
                                _learnedProfile = null;
                                _bannedNoiseZonesNorm.clear();
                              });
                              setDialogState(() {});
                            },
                            child: const Text(
                              'Сбросить',
                              style: TextStyle(fontSize: 11),
                            ),
                          ),
                        ],
                      ),
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

        double sumLum = 0.0;
        double sumLumSq = 0.0;
        int count = 0;

        final double cx = width.toDouble() * _gridCenterXNorm;
        final double cy = height.toDouble() * _gridCenterYNorm;
        final double maxRadiusSq = math
            .pow(math.min(width, height).toDouble() * _gridRadiusNorm * 0.9, 2)
            .toDouble();

        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final double dx = x.toDouble() - cx;
            final double dy = y.toDouble() - cy;
            if (dx * dx + dy * dy <= maxRadiusSq) {
              final int idx = (y * width + x) * 4;
              final int r = pixels[idx];
              final int g = pixels[idx + 1];
              final int b = pixels[idx + 2];
              final double lum = 0.299 * r + 0.587 * g + 0.114 * b;

              if (lum > 15.0) {
                sumLum += lum;
                sumLumSq += lum * lum;
                count++;
              }
            }
          }
        }

        if (count > 50) {
          final double meanLum = sumLum / count.toDouble();
          final double variance =
              (sumLumSq / count.toDouble()) - (meanLum * meanLum);
          final double stdDev = math.sqrt(math.max(0.0, variance));

          final double targetBrightness = ((210.0 - meanLum) / 35.0).clamp(
            -1.5,
            1.5,
          );
          final double targetSharpness = ((42.0 - stdDev) / 12.0).clamp(
            1.0,
            3.5,
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

  Offset _screenToNormalizedPhotoCoords(Offset localPos, Size viewSize) {
    final Offset center = Offset(viewSize.width / 2.0, viewSize.height / 2.0);
    final Offset unpanned = localPos - _photoPanOffset;
    final Offset unscaled = center + (unpanned - center) / _currentZoom;
    return Offset(
      (unscaled.dx / viewSize.width).clamp(0.0, 1.0),
      (unscaled.dy / viewSize.height).clamp(0.0, 1.0),
    );
  }

  int _findHitCapsuleIndex(
    double tapXNorm,
    double tapYNorm,
    Size viewSize, {
    double hitScale = 1.45,
  }) {
    final double aspect = viewSize.height / viewSize.width;
    int bestIdx = -1;
    double bestDist = double.infinity;

    for (int i = 0; i < _capsules.length; i++) {
      final c = _capsules[i];
      final double dx = tapXNorm - c.cxNorm;
      final double dy = (tapYNorm - c.cyNorm) * aspect;
      final double dist = math.sqrt(dx * dx + dy * dy);
      final double hitRadius = math.max(
        c.rNorm * hitScale,
        0.018 / math.sqrt(_currentZoom),
      );
      if (dist <= hitRadius && dist < bestDist) {
        bestDist = dist;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  void _onScaleStart(ScaleStartDetails details, Size viewSize) {
    FocusManager.instance.primaryFocus?.unfocus();
    _baseZoomOnScaleStart = _currentZoom;
    _baseGridRadiusOnScaleStart = _gridRadiusNorm;
    _draggingCapsuleIndex = null;

    if (_manualEditCapsules &&
        !_adjustGridMode &&
        _loadedImageBytes != null &&
        details.pointerCount == 1) {
      final Offset normPos = _screenToNormalizedPhotoCoords(
        details.localFocalPoint,
        viewSize,
      );
      final int hitIdx = _findHitCapsuleIndex(
        normPos.dx,
        normPos.dy,
        viewSize,
        hitScale: 1.6,
      );
      if (hitIdx >= 0) {
        setState(() {
          _draggingCapsuleIndex = hitIdx;
          _selectedCapsuleIndex = hitIdx;
          _manualBrushRNorm = _capsules[hitIdx].rNorm;
        });
      }
    }
  }

  void _onScaleUpdate(ScaleUpdateDetails details, Size viewSize) {
    if (_adjustGridMode) {
      setState(() {
        _gridCenterXNorm =
            (_gridCenterXNorm +
                    details.focalPointDelta.dx /
                        (viewSize.width * _currentZoom))
                .clamp(0.12, 0.88);
        _gridCenterYNorm =
            (_gridCenterYNorm +
                    details.focalPointDelta.dy /
                        (viewSize.height * _currentZoom))
                .clamp(0.12, 0.88);

        if (details.pointerCount >= 2) {
          _gridRadiusNorm = (_baseGridRadiusOnScaleStart * details.scale).clamp(
            0.12,
            0.48,
          );
        }
      });
      return;
    }

    if (details.pointerCount >= 2) {
      _draggingCapsuleIndex = null;
      final double newZoom = (_baseZoomOnScaleStart * details.scale).clamp(
        _minZoom,
        _maxZoom,
      );
      _updateZoom(newZoom);
      return;
    }

    if (_manualEditCapsules &&
        _draggingCapsuleIndex != null &&
        _loadedImageBytes != null) {
      final int idx = _draggingCapsuleIndex!;
      if (idx >= 0 && idx < _capsules.length) {
        final Offset normPos = _screenToNormalizedPhotoCoords(
          details.localFocalPoint,
          viewSize,
        );
        final DetectedCapsule snapped = _createCapsuleWithMagneticSnap(
          normPos.dx,
          normPos.dy,
          ignoreIndex: idx,
          preserveRNorm: _capsules[idx].rNorm,
        );
        setState(() {
          _capsules[idx] = snapped;
          _selectedCapsuleIndex = idx;
        });
      }
      return;
    }

    if (_loadedImageBytes != null && _currentZoom > 1.02) {
      final double maxDx = (viewSize.width * (_currentZoom - 1.0)) / 2.0;
      final double maxDy = (viewSize.height * (_currentZoom - 1.0)) / 2.0;
      setState(() {
        _photoPanOffset = Offset(
          (_photoPanOffset.dx + details.focalPointDelta.dx).clamp(
            -maxDx,
            maxDx,
          ),
          (_photoPanOffset.dy + details.focalPointDelta.dy).clamp(
            -maxDy,
            maxDy,
          ),
        );
      });
    }
  }

  void _onScaleEnd(ScaleEndDetails details) {
    if (_draggingCapsuleIndex != null) {
      setState(() {
        _draggingCapsuleIndex = null;
      });
    }
  }

  Future<void> _onTapViewport(TapUpDetails details, Size viewSize) async {
    if (FocusScope.of(context).hasFocus) {
      FocusScope.of(context).unfocus();
      return;
    }

    if (_adjustGridMode) return;

    final Offset normTap = _screenToNormalizedPhotoCoords(
      details.localPosition,
      viewSize,
    );
    final double tapXNorm = normTap.dx;
    final double tapYNorm = normTap.dy;

    if (_manualEditCapsules && _loadedImageBytes != null) {
      final int hitIndex = _findHitCapsuleIndex(tapXNorm, tapYNorm, viewSize);

      setState(() {
        _draggingCapsuleIndex = null;
        if (hitIndex >= 0) {
          final removed = _capsules.removeAt(hitIndex);
          _bannedNoiseZonesNorm.add(Offset(removed.cxNorm, removed.cyNorm));
          if (_selectedCapsuleIndex == hitIndex) {
            _selectedCapsuleIndex = _capsules.isNotEmpty
                ? _capsules.length - 1
                : null;
          } else if (_selectedCapsuleIndex != null &&
              _selectedCapsuleIndex! > hitIndex) {
            _selectedCapsuleIndex = _selectedCapsuleIndex! - 1;
          }
        } else {
          _bannedNoiseZonesNorm.removeWhere(
            (bz) =>
                (bz.dx - tapXNorm).abs() < 0.025 &&
                (bz.dy - tapYNorm).abs() < 0.025,
          );

          final DetectedCapsule newCap = _createCapsuleWithMagneticSnap(
            tapXNorm,
            tapYNorm,
          );
          _capsules.add(newCap);
          _selectedCapsuleIndex = _capsules.length - 1;
          _manualBrushRNorm = newCap.rNorm;
        }
      });
      return;
    }

    if (_loadedImageBytes == null &&
        _controller != null &&
        _isCameraInitialized) {
      setState(() {
        _focusPoint = details.localPosition;
      });

      try {
        final offset = Offset(
          (details.localPosition.dx / viewSize.width).clamp(0.0, 1.0),
          (details.localPosition.dy / viewSize.height).clamp(0.0, 1.0),
        );
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

  Future<void> _toggleManualEditMode() async {
    FocusManager.instance.primaryFocus?.unfocus();
    if (_loadedImageBytes == null) {
      await _obtainCurrentImageBytes();
    }
    if (_loadedImageBytes == null) return;

    setState(() {
      _manualEditCapsules = !_manualEditCapsules;
      if (_manualEditCapsules) {
        _adjustGridMode = false;
        _showCapsuleOverlay = true;
        if (_capsules.isNotEmpty) {
          _selectedCapsuleIndex ??= _capsules.length - 1;
          _manualBrushRNorm = _capsules[_selectedCapsuleIndex!].rNorm;
        }
      }
    });
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

  Future<Uint8List> _processAndStampImage(Uint8List inputBytes) async {
    final ui.Codec codec = await ui.instantiateImageCodec(inputBytes);
    final ui.FrameInfo frame = await codec.getNextFrame();
    final ui.Image srcImage = frame.image;

    final int w = srcImage.width;
    final int h = srcImage.height;
    final double wD = w.toDouble();
    final double hD = h.toDouble();

    final ui.PictureRecorder recorder = ui.PictureRecorder();
    final Canvas canvas = Canvas(recorder, Rect.fromLTWH(0, 0, wD, hD));

    final Paint imagePaint = Paint()
      ..colorFilter = _buildColorFilter(isFinalCapture: true)
      ..filterQuality = FilterQuality.high;
    canvas.drawImage(srcImage, Offset.zero, imagePaint);

    final double scaleUm = _getScaleMicrons();

    if (_showCapsuleOverlay && _capsules.isNotEmpty) {
      final Paint circlePaint = Paint()
        ..color = Colors.greenAccent
        ..style = PaintingStyle.stroke
        ..strokeWidth = math.max(1.2, wD * 0.0018);

      final Paint centerPaint = Paint()
        ..color = Colors.amberAccent
        ..style = PaintingStyle.fill;

      for (int i = 0; i < _capsules.length; i++) {
        final c = _capsules[i];
        final double cx = c.cxNorm * wD;
        final double cy = c.cyNorm * hD;
        final double r = c.rNorm * wD;
        final double diamUm = c.diameterMicrons(scaleUm);

        canvas.drawCircle(Offset(cx, cy), r, circlePaint);
        canvas.drawCircle(
          Offset(cx, cy),
          math.max(1.0, wD * 0.0014),
          centerPaint,
        );

        final TextPainter tp = TextPainter(
          text: TextSpan(
            text: '${diamUm.toStringAsFixed(0)} мкм',
            style: TextStyle(
              color: Colors.yellowAccent,
              fontSize: math.max(6.0, wD * 0.0095),
              fontWeight: FontWeight.bold,
              backgroundColor: const Color.fromRGBO(0, 0, 0, 0.45),
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout();

        tp.paint(canvas, Offset(cx - tp.width / 2.0, cy - r - tp.height - 1.0));
      }
    }

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
        ? 'Капсул: ${_capsules.length} шт.$avgText  |  Ø ср: ${stats['mean']!.toStringAsFixed(1)} мкм [${stats['min']!.toStringAsFixed(0)}–${stats['max']!.toStringAsFixed(0)} мкм]'
        : 'Капсулы: подсчет не проводился';
    final String line3Params =
        'Фильтр: ${_selectedFilter.title}  |  Ярк: ${_brightness.toStringAsFixed(1)}  |  Четк: ${_sharpness.toStringAsFixed(1)}';
    final String line4Date = 'Дата съемки: $dateStr';

    final double textMaxWidth = wD * 0.70;
    final double titleFontSize = wD * 0.033;
    final double subFontSize = wD * 0.023;

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

    final double verticalPadding = hD * 0.015;
    final double lineSpacing = hD * 0.005;
    final double totalTextHeight =
        tp1.height +
        tp2.height +
        tp3.height +
        tp4.height +
        (lineSpacing * 3.0) +
        (verticalPadding * 2.2);

    final double barHeight = math.max(hD * 0.115, totalTextHeight);
    final double barTop = hD - barHeight;

    canvas.drawRect(
      Rect.fromLTWH(0, barTop, wD, barHeight),
      Paint()..color = const Color.fromRGBO(0, 0, 0, 0.82),
    );

    double currentY = barTop + verticalPadding;
    tp1.paint(canvas, Offset(wD * 0.03, currentY));
    currentY += tp1.height + lineSpacing;

    tp2.paint(canvas, Offset(wD * 0.03, currentY));
    currentY += tp2.height + lineSpacing;

    tp3.paint(canvas, Offset(wD * 0.03, currentY));
    currentY += tp3.height + lineSpacing;

    tp4.paint(canvas, Offset(wD * 0.03, currentY));

    final String scaleLabel = '${scaleUm.toInt()} мкм';
    final double scaleBarWidth = wD * 0.20;
    final double scaleRight = wD * 0.96;
    final double scaleLeft = scaleRight - scaleBarWidth;
    final double scaleY = barTop + (barHeight * 0.65);

    final Paint scalePaint = Paint()
      ..color = Colors.cyanAccent
      ..strokeWidth = math.max(3.0, hD * 0.003);

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
        scaleLeft + (scaleBarWidth - scaleTextPainter.width) / 2.0,
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
    final double currentBrushDiamUm =
        2.0 * _manualBrushRNorm * (scaleUm / 0.20);

    return Scaffold(
      resizeToAvoidBottomInset: true,
      body: SafeArea(
        child: Column(
          children: [
            // 1. ВЕРХНЯЯ ПАНЕЛЬ
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
                    tooltip: 'Показать / скрыть сетку',
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

            // 2. ПАНЕЛЬ СТАТИСТИКИ, ПЕРЕКЛЮЧАТЕЛЯ НЕЙРОСЕТИ (ИИ) И ПРАВКИ
            Container(
              width: double.infinity,
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              color: Colors.grey.shade900,
              child: Row(
                children: [
                  Icon(
                    _adjustGridMode
                        ? Icons.open_with
                        : (_manualEditCapsules
                              ? Icons.touch_app
                              : Icons.bubble_chart),
                    size: 16,
                    color: _adjustGridMode
                        ? Colors.cyanAccent
                        : (_manualEditCapsules
                              ? Colors.amberAccent
                              : Colors.greenAccent),
                  ),
                  const SizedBox(width: 5),
                  Expanded(
                    child: Text(
                      _adjustGridMode
                          ? '1 палец — двигать сетку, 2 пальца — размер круга'
                          : (_manualEditCapsules
                                ? 'Тап: +/– | Справа: Ø (${_capsules.length} шт., Ø ${diameterStats['mean']!.toStringAsFixed(1)} мкм)'
                                : (_capsules.isEmpty
                                      ? (isPhotoMode
                                            ? 'Нажмите «Считать», «Мульти» или «Правка»'
                                            : 'Зум до 20x. Нажмите «Галерея» или «Считать»')
                                      : 'Капсул: ${_capsules.length} шт.${_multiPassAverageCount != null ? " (Ср: ${_multiPassAverageCount!.toStringAsFixed(1)})" : ""} | '
                                            'Ø: ${diameterStats['mean']!.toStringAsFixed(1)} мкм')),
                      style: TextStyle(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w600,
                        color: _adjustGridMode
                            ? Colors.cyanAccent
                            : (_manualEditCapsules
                                  ? Colors.amberAccent
                                  : (_capsules.isEmpty
                                        ? Colors.white70
                                        : Colors.greenAccent)),
                      ),
                    ),
                  ),
                  if (_isTfliteLoaded && !_manualEditCapsules) ...[
                    InkWell(
                      onTap: () => setState(() => _useAiModel = !_useAiModel),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 5,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: _useAiModel
                              ? Colors.greenAccent.withOpacity(0.25)
                              : Colors.white12,
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                            color: _useAiModel
                                ? Colors.greenAccent
                                : Colors.white24,
                          ),
                        ),
                        child: Text(
                          _useAiModel ? 'ИИ: ВКЛ' : 'Алгоритм',
                          style: TextStyle(
                            fontSize: 9.5,
                            fontWeight: FontWeight.bold,
                            color: _useAiModel
                                ? Colors.greenAccent
                                : Colors.white70,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                  ],
                  InkWell(
                    onTap: () => setState(() {
                      _adjustGridMode = !_adjustGridMode;
                      if (_adjustGridMode) {
                        _showGrid = true;
                        _manualEditCapsules = false;
                      }
                    }),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 5,
                        vertical: 3,
                      ),
                      decoration: BoxDecoration(
                        color: _adjustGridMode
                            ? Colors.cyanAccent
                            : Colors.white12,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        _adjustGridMode ? 'Сетка: ОК' : 'Сетка',
                        style: TextStyle(
                          fontSize: 9.8,
                          fontWeight: FontWeight.bold,
                          color: _adjustGridMode
                              ? Colors.black
                              : Colors.cyanAccent,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 4),
                  InkWell(
                    onTap: _toggleManualEditMode,
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 5,
                        vertical: 3,
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
                          fontSize: 9.8,
                          fontWeight: FontWeight.bold,
                          color: _manualEditCapsules
                              ? Colors.black
                              : Colors.white70,
                        ),
                      ),
                    ),
                  ),
                  if (_manualEditCapsules) ...[
                    const SizedBox(width: 4),
                    InkWell(
                      onTap: () => setState(
                        () => _magneticSnapEnabled = !_magneticSnapEnabled,
                      ),
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 5,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: _magneticSnapEnabled
                              ? Colors.greenAccent.withOpacity(0.25)
                              : Colors.white10,
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(
                            color: _magneticSnapEnabled
                                ? Colors.greenAccent
                                : Colors.white24,
                          ),
                        ),
                        child: Text(
                          _magneticSnapEnabled ? 'Магнит' : 'Своб.',
                          style: TextStyle(
                            fontSize: 9.5,
                            fontWeight: FontWeight.bold,
                            color: _magneticSnapEnabled
                                ? Colors.greenAccent
                                : Colors.white54,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(width: 4),
                    InkWell(
                      onTap: _isProcessing ? null : _trainFromMarkedCapsules,
                      child: Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 5,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.purpleAccent.withOpacity(0.28),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: Colors.purpleAccent),
                        ),
                        child: Text(
                          _learnedProfile != null
                              ? 'Обучить (${_learnedProfile!.trainingCount})'
                              : 'Обучить',
                          style: const TextStyle(
                            fontSize: 9.5,
                            fontWeight: FontWeight.bold,
                            color: Colors.purpleAccent,
                          ),
                        ),
                      ),
                    ),
                    if (_capsules.isNotEmpty) ...[
                      const SizedBox(width: 4),
                      InkWell(
                        onTap: () => setState(() {
                          _capsules.clear();
                          _selectedCapsuleIndex = null;
                        }),
                        child: const Icon(
                          Icons.delete_sweep,
                          size: 17,
                          color: Colors.redAccent,
                        ),
                      ),
                    ],
                  ],
                  if (isPhotoMode) ...[
                    const SizedBox(width: 4),
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

            // 3. ОБЛАСТЬ ИЗОБРАЖЕНИЯ + БОКОВОЙ ПОЛЗУНОК ДИАМЕТРА КРУЖКА
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
                    final Size viewSize = Size(viewWidth, viewHeight);

                    return SizedBox(
                      width: viewWidth,
                      height: viewHeight,
                      child: ClipRect(
                        child: Stack(
                          alignment: Alignment.center,
                          children: [
                            GestureDetector(
                              behavior: HitTestBehavior.opaque,
                              onTapUp: (details) =>
                                  _onTapViewport(details, viewSize),
                              onScaleStart: (details) =>
                                  _onScaleStart(details, viewSize),
                              onScaleUpdate: (details) =>
                                  _onScaleUpdate(details, viewSize),
                              onScaleEnd: _onScaleEnd,
                              child: Stack(
                                alignment: Alignment.center,
                                children: [
                                  if (isPhotoMode)
                                    Transform.translate(
                                      offset: _photoPanOffset,
                                      child: Transform.scale(
                                        scale: _currentZoom,
                                        child: SizedBox(
                                          width: viewWidth,
                                          height: viewHeight,
                                          child: Stack(
                                            fit: StackFit.expand,
                                            children: [
                                              ColorFiltered(
                                                colorFilter:
                                                    _buildColorFilter(),
                                                child: Image.memory(
                                                  _loadedImageBytes!,
                                                  fit: BoxFit.fill,
                                                  gaplessPlayback: true,
                                                ),
                                              ),
                                              if (_showCapsuleOverlay &&
                                                  _capsules.isNotEmpty)
                                                IgnorePointer(
                                                  child: CustomPaint(
                                                    size: viewSize,
                                                    painter: CapsuleOverlayPainter(
                                                      capsules: _capsules,
                                                      scaleMicrons: scaleUm,
                                                      zoomLevel: _currentZoom,
                                                      activeIndex:
                                                          _selectedCapsuleIndex ??
                                                          _draggingCapsuleIndex,
                                                      isEditMode:
                                                          _manualEditCapsules,
                                                    ),
                                                  ),
                                                ),
                                              if (_showGrid)
                                                IgnorePointer(
                                                  child: CustomPaint(
                                                    size: viewSize,
                                                    painter:
                                                        MicroscopeGridPainter(
                                                          centerXNorm:
                                                              _gridCenterXNorm,
                                                          centerYNorm:
                                                              _gridCenterYNorm,
                                                          radiusNorm:
                                                              _gridRadiusNorm,
                                                          isAdjusting:
                                                              _adjustGridMode,
                                                          zoomLevel:
                                                              _currentZoom,
                                                        ),
                                                  ),
                                                ),
                                            ],
                                          ),
                                        ),
                                      ),
                                    )
                                  else if (_isCameraInitialized &&
                                      _controller != null)
                                    Stack(
                                      fit: StackFit.expand,
                                      children: [
                                        FittedBox(
                                          fit: BoxFit.cover,
                                          child: SizedBox(
                                            width: viewWidth,
                                            height:
                                                viewWidth *
                                                _controller!.value.aspectRatio,
                                            child: ColorFiltered(
                                              colorFilter: _buildColorFilter(),
                                              child: CameraPreview(
                                                _controller!,
                                              ),
                                            ),
                                          ),
                                        ),
                                        if (_showCapsuleOverlay &&
                                            _capsules.isNotEmpty)
                                          IgnorePointer(
                                            child: CustomPaint(
                                              size: viewSize,
                                              painter: CapsuleOverlayPainter(
                                                capsules: _capsules,
                                                scaleMicrons: scaleUm,
                                                zoomLevel: _currentZoom,
                                                activeIndex:
                                                    _selectedCapsuleIndex ??
                                                    _draggingCapsuleIndex,
                                                isEditMode: _manualEditCapsules,
                                              ),
                                            ),
                                          ),
                                        if (_showGrid)
                                          IgnorePointer(
                                            child: CustomPaint(
                                              size: viewSize,
                                              painter: MicroscopeGridPainter(
                                                centerXNorm: _gridCenterXNorm,
                                                centerYNorm: _gridCenterYNorm,
                                                radiusNorm: _gridRadiusNorm,
                                                isAdjusting: _adjustGridMode,
                                                zoomLevel: 1.0,
                                              ),
                                            ),
                                          ),
                                      ],
                                    )
                                  else
                                    Column(
                                      mainAxisAlignment:
                                          MainAxisAlignment.center,
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
                                          borderRadius: BorderRadius.circular(
                                            4,
                                          ),
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

                            // БОКОВОЙ ПОЛЗУНОК ДИАМЕТРА КРУЖКА КАПСУЛЫ В РЕЖИМЕ «ПРАВКА»
                            if (_manualEditCapsules)
                              Positioned(
                                right: 8,
                                top: 20,
                                bottom: 20,
                                child: Container(
                                  width: 46,
                                  padding: const EdgeInsets.symmetric(
                                    vertical: 8,
                                    horizontal: 4,
                                  ),
                                  decoration: BoxDecoration(
                                    color: const Color.fromRGBO(
                                      18,
                                      18,
                                      18,
                                      0.86,
                                    ),
                                    borderRadius: BorderRadius.circular(24),
                                    border: Border.all(
                                      color: Colors.cyanAccent.withOpacity(0.6),
                                      width: 1.2,
                                    ),
                                  ),
                                  child: Column(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: [
                                      const Text(
                                        'Ø круга',
                                        textAlign: TextAlign.center,
                                        style: TextStyle(
                                          fontSize: 8.5,
                                          fontWeight: FontWeight.bold,
                                          color: Colors.cyanAccent,
                                        ),
                                      ),
                                      Text(
                                        '${currentBrushDiamUm.toStringAsFixed(1)}\nмкм',
                                        textAlign: TextAlign.center,
                                        style: const TextStyle(
                                          fontSize: 9.0,
                                          fontWeight: FontWeight.bold,
                                          color: Colors.amberAccent,
                                        ),
                                      ),
                                      InkWell(
                                        onTap: () =>
                                            _updateManualCircleRadiusNorm(
                                              _manualBrushRNorm + 0.0012,
                                            ),
                                        child: Container(
                                          padding: const EdgeInsets.all(4),
                                          decoration: const BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: Colors.white12,
                                          ),
                                          child: const Icon(
                                            Icons.add_circle_outline,
                                            size: 18,
                                            color: Colors.cyanAccent,
                                          ),
                                        ),
                                      ),
                                      Expanded(
                                        child: RotatedBox(
                                          quarterTurns: 3,
                                          child: SliderTheme(
                                            data: SliderTheme.of(context).copyWith(
                                              trackHeight: 3.0,
                                              thumbShape:
                                                  const RoundSliderThumbShape(
                                                    enabledThumbRadius: 7.0,
                                                  ),
                                              overlayShape:
                                                  const RoundSliderOverlayShape(
                                                    overlayRadius: 12.0,
                                                  ),
                                            ),
                                            child: Slider(
                                              value: _manualBrushRNorm.clamp(
                                                0.003,
                                                0.055,
                                              ),
                                              min: 0.003,
                                              max: 0.055,
                                              activeColor: Colors.cyanAccent,
                                              inactiveColor: Colors.white24,
                                              onChanged:
                                                  _updateManualCircleRadiusNorm,
                                            ),
                                          ),
                                        ),
                                      ),
                                      InkWell(
                                        onTap: () =>
                                            _updateManualCircleRadiusNorm(
                                              _manualBrushRNorm - 0.0012,
                                            ),
                                        child: Container(
                                          padding: const EdgeInsets.all(4),
                                          decoration: const BoxDecoration(
                                            shape: BoxShape.circle,
                                            color: Colors.white12,
                                          ),
                                          child: const Icon(
                                            Icons.remove_circle_outline,
                                            size: 18,
                                            color: Colors.cyanAccent,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),

            // 4. НИЖНЯЯ ПАНЕЛЬ УПРАВЛЕНИЯ
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
              color: Colors.black87,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
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
                          onSelected: (_) => setState(() {
                            _selectedFilter = filter;
                            _snapBufferDirty = true;
                          }),
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 3),

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

                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
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

                      IconButton(
                        visualDensity: VisualDensity.compact,
                        tooltip: 'Сброс настроек и сетки',
                        onPressed: () {
                          FocusManager.instance.primaryFocus?.unfocus();
                          _updateBrightness(0.0);
                          _updateSharpness(1.0);
                          _updateZoom(1.0);
                          setState(() {
                            _selectedFilter = LabFilter.standard;
                            _capsules = [];
                            _bannedNoiseZonesNorm.clear();
                            _selectedCapsuleIndex = null;
                            _multiPassAverageCount = null;
                            _gridCenterXNorm = 0.5;
                            _gridCenterYNorm = 0.5;
                            _gridRadiusNorm = 0.36;
                            _adjustGridMode = false;
                            _manualEditCapsules = false;
                            _minCapsuleRadiusPx = 4;
                            _maxCapsuleRadiusPx = 14;
                            _detectionSensitivity = 0.60;
                            _snapBufferDirty = true;
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

class CapsuleOverlayPainter extends CustomPainter {
  final List<DetectedCapsule> capsules;
  final double scaleMicrons;
  final double zoomLevel;
  final int? activeIndex;
  final bool isEditMode;

  CapsuleOverlayPainter({
    required this.capsules,
    required this.scaleMicrons,
    this.zoomLevel = 1.0,
    this.activeIndex,
    this.isEditMode = false,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double z = zoomLevel.clamp(1.0, 25.0);
    final double textShrinkFactor = math.pow(z, 1.55).toDouble();
    final double strokeShrinkFactor = math.pow(z, 1.25).toDouble();

    final Paint circlePaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = (1.3 / strokeShrinkFactor).clamp(0.12, 1.5);

    final Paint activeCirclePaint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = (2.0 / strokeShrinkFactor).clamp(0.20, 2.2);

    final Paint centerPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.fill;

    final double dotRadius = ((isEditMode ? 2.0 : 1.4) / strokeShrinkFactor)
        .clamp(0.15, 2.2);
    final double dynamicFontSize = (6.8 / textShrinkFactor).clamp(0.55, 7.0);
    final double labelOffset = (1.4 / textShrinkFactor).clamp(0.1, 2.0);

    for (int i = 0; i < capsules.length; i++) {
      final c = capsules[i];
      final bool isActive = i == activeIndex;
      final double cx = c.cxNorm * size.width;
      final double cy = c.cyNorm * size.height;
      final double r = c.rNorm * size.width;
      final double diamUm = c.diameterMicrons(scaleMicrons);

      canvas.drawCircle(
        Offset(cx, cy),
        r,
        isActive ? activeCirclePaint : circlePaint,
      );
      canvas.drawCircle(Offset(cx, cy), dotRadius, centerPaint);

      final TextPainter tp = TextPainter(
        text: TextSpan(
          text: '${diamUm.toStringAsFixed(0)} мкм',
          style: TextStyle(
            color: isActive ? Colors.cyanAccent : Colors.yellowAccent,
            fontSize: dynamicFontSize,
            fontWeight: FontWeight.bold,
            backgroundColor: const Color.fromRGBO(0, 0, 0, 0.45),
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();

      tp.paint(
        canvas,
        Offset(cx - tp.width / 2.0, cy - r - tp.height - labelOffset),
      );
    }
  }

  @override
  bool shouldRepaint(covariant CapsuleOverlayPainter oldDelegate) {
    return oldDelegate.zoomLevel != zoomLevel ||
        oldDelegate.capsules != capsules ||
        oldDelegate.activeIndex != activeIndex ||
        oldDelegate.isEditMode != isEditMode ||
        oldDelegate.scaleMicrons != scaleMicrons;
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
    final double? parsed = double.tryParse(normalized);

    if (parsed != null) {
      final double clamped = parsed.clamp(widget.min, widget.max);
      final double rounded = double.parse(clamped.toStringAsFixed(1));
      _textController.text = rounded.toStringAsFixed(1);
      widget.onChanged(rounded);
    } else {
      _textController.text = widget.value.toStringAsFixed(1);
    }
  }

  void _stepChange(double delta) {
    FocusManager.instance.primaryFocus?.unfocus();
    final double nextVal = (widget.value + delta).clamp(widget.min, widget.max);
    final double rounded = double.parse(nextVal.toStringAsFixed(1));
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
  final double centerXNorm;
  final double centerYNorm;
  final double radiusNorm;
  final bool isAdjusting;
  final double zoomLevel;

  MicroscopeGridPainter({
    this.centerXNorm = 0.5,
    this.centerYNorm = 0.5,
    this.radiusNorm = 0.36,
    this.isAdjusting = false,
    this.zoomLevel = 1.0,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final double z = zoomLevel.clamp(1.0, 25.0);
    final Paint paint = Paint()
      ..color = isAdjusting
          ? const Color.fromRGBO(255, 215, 64, 0.85)
          : const Color.fromRGBO(0, 255, 255, 0.45)
      ..strokeWidth = (isAdjusting ? 2.2 : 1.2) / z
      ..style = PaintingStyle.stroke;

    final double cx = size.width * centerXNorm;
    final double cy = size.height * centerYNorm;
    final double radius = math.min(size.width, size.height) * radiusNorm;

    canvas.drawCircle(Offset(cx, cy), radius, paint);
    canvas.drawLine(Offset(cx - radius, cy), Offset(cx + radius, cy), paint);
    canvas.drawLine(Offset(cx, cy - radius), Offset(cx, cy + radius), paint);

    final double step = radius / 5.0;
    final double tick = 4.0 / z;
    for (int i = -4; i <= 4; i++) {
      if (i == 0) continue;
      final double d = i.toDouble() * step;
      canvas.drawLine(
        Offset(cx + d, cy - tick),
        Offset(cx + d, cy + tick),
        paint,
      );
      canvas.drawLine(
        Offset(cx - tick, cy + d),
        Offset(cx + tick, cy + d),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant MicroscopeGridPainter oldDelegate) {
    return oldDelegate.centerXNorm != centerXNorm ||
        oldDelegate.centerYNorm != centerYNorm ||
        oldDelegate.radiusNorm != radiusNorm ||
        oldDelegate.isAdjusting != isAdjusting ||
        oldDelegate.zoomLevel != zoomLevel;
  }
}
