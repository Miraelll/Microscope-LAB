<div align="center">
  
# 🔬 Microscope LAB

## Microscope photography and microcapsule counting on your phone: manual image controls, optical filters, hybrid AI + contour detection, trainable profiles, diameters in μm, and a stamped, scaled result saved to your gallery.

[![Flutter](https://img.shields.io/badge/Flutter-cross--platform-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Dart](https://img.shields.io/badge/Dart-3.13.5%2B-0175C2?logo=dart&logoColor=white)](https://dart.dev/)
[![AI](https://img.shields.io/badge/AI-YOLOv8%20TFLite-7B1FA2)](#ai-model-optional)
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS-333333)](#platform-support)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Contents

- [Overview](#overview)
- [What's new](#whats-new)
- [Features](#features)
- [How capsule counting works](#how-capsule-counting-works)
- [Scale ruler and diameters: read this first](#scale-ruler-and-diameters-read-this-first)
- [AI model (optional)](#ai-model-optional)
- [Platform support](#platform-support)
- [Getting started](#getting-started)
- [Building a release](#building-a-release)
- [How to use](#how-to-use)
- [The saved image](#the-saved-image)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Permissions and privacy](#permissions-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

**Microscope LAB** is a Flutter application for digital microscopy. Put a phone camera at the eyepiece or camera adapter of a microscope (or open an existing photo from the gallery), tune the image with exact numeric controls, choose an optical filter and a magnification, and let the app find round, ring-shaped objects such as microcapsules, count them, and estimate their diameters in micrometres (μm).

Detection combines two engines: a contour and morphology algorithm that works out of the box, and an optional YOLOv8 TFLite neural network whose results are merged with the algorithm's. You can always correct the result by hand, and teach the algorithm what your capsules look like from the ones you confirmed.

The result is saved as a PNG with the filter applied, every detected capsule outlined and labeled with its diameter, a scale ruler, and an information stamp with the count and diameter statistics.

> **Note:** the app interface is currently in Russian. Button names are given below in English with the original label in «quotes».
> **Important:** diameters and the scale ruler are estimates derived from the selected magnification, not a calibrated measurement. See [Scale ruler and diameters: read this first](#scale-ruler-and-diameters-read-this-first).

## What's new

- **Hybrid detection.** An optional YOLOv8 TFLite model runs alongside the contour algorithm; the algorithm fills in capsules the network missed. If no model is bundled, the app silently uses the algorithm alone.
- **New contour algorithm** built for transparent, ring-walled capsules: local background subtraction, a 12-ray ring search with an elastic radius, and support for touching capsules in clusters and figure-eights. It replaces the previous circle Hough transform.
- **Fiber and debris rejection.** Long fibers, sticks, and sharp black dust specks are recognized by their shape (elongation from second-order moments) and excluded together with a small halo.
- **Trainable profile («Обучить» / Train).** Confirm a few real capsules by hand, tap Train, and the detector adapts its wall-signal threshold and radius range to your sample. The profile is stored on the device and can be reset.
- **Persistent manual corrections.** A capsule you delete is remembered as a banned zone, so the next run does not bring it back. Manually added or confirmed capsules are kept when detection is repeated.
- **Magnetic snap.** Added or dragged markers snap to the nearest ring contour and adopt its radius (can be switched to free placement).
- **Higher working resolution:** analysis runs at a 640 px working width (previously 320 px).
- **Interactive ocular grid** that auto-fits to the bright microscope circle on loaded photos and defines the region of interest (ROI).
- **Pinch-to-zoom up to 20×** with panning in photo mode; labels and contour lines shrink as you zoom so they do not clutter the view.
- **Multi-pass calculation** over several brightness and sharpness values with an averaged count.

## Features

### Live camera and photo mode

- Live preview from the rear camera (the first available camera is used if there is no rear one), portrait orientation.
- Tap the preview to set focus and exposure; focus is then locked. The focus button («Фокус» / «БЛОК») switches between locked and automatic focus.
- Two-finger pinch zoom up to 20×; in photo mode, drag to pan when zoomed in.
- Open an existing photo from the gallery and work with it exactly like a live frame; the «Камера» button returns you to the live camera.
- Pressing a counting or edit button in live mode first grabs a frame from the camera and switches to photo mode, so detection and editing always run on a still image.
- The reset button restores default brightness, sharpness, zoom, filter, grid position, detection settings, and clears the current capsules. A saved training profile is kept.
Available ranges depend on the phone's camera hardware and operating system.

### Image controls

- **Brightness («Ярк»):** spinbox from -10.0 to 10.0 in 0.1 steps. In live mode it also adjusts the camera exposure offset.
- **Sharpness («Четк», contrast):** spinbox from 0.0 to 10.0 in 0.1 steps.
- **AUTO («АВТО»):** estimates brightness and sharpness from the central part of a frame.
- Values can be typed in directly or changed with the +/- buttons.
Brightness and sharpness change how the image looks and how strongly edges stand out to the detector. They cannot recover detail the sensor did not capture.

### Optical filters

| Filter (in the app)      | Purpose                                                   |
|--------------------------|-----------------------------------------------------------|
| Standard («Стандарт»)    | Natural-looking baseline image                            |
| High Detail («Контуры HD») | Increased contrast to emphasize visible edges           |
| Clean White («Чистый белый») | Cooler, slightly blue-toned image                     |
| Achromat B/W («Ахромат Ч/Б») | Monochrome-style image                                |
| Dark Field («Темное поле») | Inverted, high-contrast appearance for visual exploration |

Filters apply to the preview, to the saved image, and to the input of the AI model. The contour algorithm itself works from brightness and sharpness only. Filters are visual presets and do not replace real laboratory illumination.

### Capsule counting

- **Count («Считать»):** a single detection pass with the current brightness and sharpness.
- **Multi («Мульти»):** enter several brightness and sharpness values separated by spaces (defaults: `-0.5 0.0 0.5` and `1.0 1.5 2.0`). The app runs every combination and reports the average count; the pass closest to the average is applied.
- **Detection settings** in the Multi dialog: capsule radius range in pixels of the 640 px working image (slider 3 to 32, default 4 to 14), and sensitivity (15% to 90%, default 60%).
- **AI / Algorithm switch** («ИИ: ВКЛ» / «Алгоритм»): shown only when a TFLite model was loaded successfully.
- **Overlay:** detected capsules are outlined and labeled with their diameter on screen and in the saved image. An eye icon shows or hides the overlay in photo mode.

### Manual editing («Правка»)

Turn on edit mode to correct the result on the photo:

- **Tap a capsule** to remove it. The spot is remembered so it is not detected again.
- **Tap an empty spot** to add a capsule.
- **Drag** a capsule to move it.
- **+ / - buttons** next to the diameter indicator change the radius of the selected capsule.
- **Magnet / Free** («Магнит» / «Своб.») toggles snapping to the nearest ring contour.
- **Train** («Обучить») learns from the capsules currently on screen (see below).
- The trash icon clears all capsules.

### Training on your sample

Mark one to three (or more) real capsules, then tap **Train**. The app measures their wall signal, hollow-center contrast, and radius range, stores the profile with `shared_preferences`, and immediately re-runs detection while keeping your marked capsules. The button shows how many times the profile has been trained. **Reset** in the Multi dialog clears it.

### Magnification and labels

- Built-in magnifications: `4x`, `10x`, `20x`, `40x`, `100x`.
- Enter your own value (for example `60x`); the list is saved locally between sessions.
- Name the sample; the name appears in the stamp.

### Ocular grid

A circle with crosshair and tick marks marks the region of interest. Press **Grid** («Сетка»), then move it with one finger and resize it with two fingers. On loaded photos it is fitted to the bright microscope circle automatically. Capsules outside the circle are ignored. The grid icon in the top bar shows or hides it.

## How capsule counting works

1. **Working image.** The photo is downscaled to a 640 px working width.
2. **Region of interest.** Only the bright circle of the ocular is analyzed: pixels must be light enough and not blue-tinted, which removes the dark border and chromatic-aberration rim. The area stays inside 90% of the grid radius, keeps a safety margin from the lens edge, and excludes banned zones left by capsules you deleted.
3. **Background suppression.** A small blur is subtracted from a wide local-background blur (difference-of-blurs, in the spirit of DoG), which removes uneven lamp illumination and high-frequency noise. The positive remainder is the "dark signal" that capsule walls produce; brightness and sharpness scale it.
4. **Adaptive wall threshold** from the statistics of that signal and the sensitivity setting. A trained profile overrides it.
5. **Fiber and debris mask.** Connected dark components are analyzed with image moments. Long, thin structures and small, very sharp black specks are blocked, with a halo around them.
6. **Ring search.** For every candidate center and radius in the allowed range, 12 rays look for a wall in an elastic band of about ±22% around the radius. A candidate needs enough wall rays and hollow rays, walls in at least three of four quadrants, a strong median ring, and a quieter outer background. It is scored by hollow contrast, outer contrast, gradient strength, and ray coverage. Seven free outer directions are enough, so capsules in clusters are still found.
7. **Overlap suppression.** Candidates are sorted by score and neighbors that overlap too much are dropped. Touching capsules in chains and figure-eights are kept. At most 250 capsules are returned.
8. **Multi-pass.** Steps 3 to 7 repeat for every brightness and sharpness pair. The pass whose count is closest to the average is kept, and its brightness and sharpness are applied to the image.
9. **AI merge (if enabled).** For each pass the YOLOv8 model produces detections inside the ROI. Contour detections that do not overlap a network detection are added to them.
10. **Manual priority.** Capsules you added or confirmed are kept, and automatic detections overlapping them are discarded.
Radius settings are in pixels of the 640 px working image. Results depend on lighting, focus, and how round and well-contrasted the objects are, so check the overlay and correct it manually when needed.

## Scale ruler and diameters: read this first

The ruler is always drawn as a segment equal to 20% of the image width. Its length in μm is chosen from the magnification label:

| Magnification | Ruler length |
|---------------|--------------|
| up to 5x      | 500 μm       |
| up to 15x     | 200 μm       |
| up to 30x     | 100 μm       |
| up to 60x     | 50 μm        |
| above 60x     | 20 μm        |

A capsule diameter in μm is `2 × radius × (ruler length ÷ 0.20)`, with the radius expressed as a fraction of the image width. That makes diameters **approximate values that depend entirely on how well this table matches your real optical setup** (objective, eyepiece or adapter, camera sensor, zoom). The app does not calibrate pixels to micrometres for a specific microscope. For quantitative work, check the ruler against a stage micrometer with your own equipment and treat the result as valid only for that setup.

## AI model (optional)

The app tries to load `assets/capsule_detector.tflite` (or `capsule_detector.tflite`) at startup. **The model file is not included in this repository.** Without it the app works with the contour algorithm only, and the AI switch is hidden.

To use your own model:

1. Train and export a YOLOv8 detector (single class is enough) to TFLite.
2. Put the file at `assets/capsule_detector.tflite` and register it in `pubspec.yaml`:

```yaml
   dependencies:
     tflite_flutter: ^<current version>
 
   flutter:
     assets:
       - assets/capsule_detector.tflite
```

3. Run `flutter pub get` and rebuild.
What the app expects and handles:

- Input `[1, 3, H, W]` (NCHW) or `[1, H, W, 3]` (NHWC), RGB normalized to 0..1, in Float32, Float16, or Uint8.
- Output in the standard YOLOv8 layout, `[1, 4 + classes, anchors]` or its transpose, with box center and size either normalized or in pixels.
- The confidence threshold follows the sensitivity slider. Overlapping boxes are removed by the app.

## Platform support

| Platform                   | Status                                | Notes                                                |
|----------------------------|---------------------------------------|------------------------------------------------------|
| Android                    | Primary target                        | Android Studio and Android SDK required for building |
| iOS                        | Primary target                        | macOS and Xcode required to build and sign           |
| Windows, macOS, Linux, Web | Flutter scaffolding present, untested | Not the main focus of this project                   |

Camera access, photo-library permissions, and some camera controls behave differently between devices. Test on real hardware before distributing a release.

## Getting started

### Requirements

- [Flutter SDK](https://docs.flutter.dev/get-started/install) compatible with the Dart constraint in `pubspec.yaml` (`^3.13.5`).
- [Git](https://git-scm.com/).
- **Android:** Android Studio, Android SDK, and an emulator or device.
- **iOS:** macOS, Xcode with command-line tools, and a simulator or device.

```
flutter --version
flutter doctor
```

### Clone and install

```

git clone https://github.com/Miraelll/Microscope-LAB.git
cd Microscope-LAB
flutter pub get
```

The code imports `tflite_flutter`. Make sure it is listed in `pubspec.yaml` (see [AI model](#ai-model-optional)); the model file itself is optional.
 
### Check the project

```
flutter analyze
flutter test
```

### Run

```
flutter devices
flutter run                   # or: flutter run -d <device-id>
```

For iOS, open `ios/Runner.xcworkspace` in Xcode, configure signing, and run on a simulator or an iPhone.

## Building a release

**Android**

```
flutter build apk --release          # build/app/outputs/flutter-apk/app-release.apk
flutter build appbundle --release    # for Google Play
```

**iOS** (macOS and Xcode required)

```
flutter build ios --release
```

Before publishing, configure the application ID or bundle identifier, signing, version, icons, and store metadata. A `codemagic.yaml` is included for CI builds with Codemagic.

## How to use

1. **Launch the app** and allow camera access.
2. **Get an image.** Attach the phone camera to the microscope, or open a photo from the gallery.
3. **Name the sample** and **choose the magnification** in the top bar.
4. **Align the grid:** on photos it fits itself to the ocular circle; adjust it with **Grid** (1 finger to move, 2 fingers to resize).
5. **Tune the image:** brightness and sharpness spinboxes (or **AUTO**), filter, zoom, focus.
6. **Count capsules:** press **Count** for one pass, or **Multi** to set several brightness and sharpness values, the radius range, and sensitivity.
7. **Review the overlay.** Turn on **Edit** to remove false detections, add missed capsules, drag markers, or change radii.
8. **Optionally train** on the capsules you confirmed so the next run fits your sample better.
9. **Save.** Press the shutter button; the stamped result goes to your photo library.
For repeatable work, keep illumination, camera position, and optical settings the same between shots.

## The saved image

Each save produces a PNG named `capsules_<timestamp>.png` containing:

- the photo with the selected filter, brightness, and sharpness applied;
- every detected capsule outlined, with a center dot and its diameter in μm;
- a dark information strip at the bottom with four lines: sample name and magnification; capsule count (and the multi-pass average) with mean diameter and the min to max range; filter, brightness, and sharpness; and the capture date and time;
- the scale ruler in the bottom-right corner of the strip.

## Project structure

```
Microscope-LAB/
├── android/                 # Android application and native configuration
├── ios/                     # iOS application and Xcode configuration
├── assets/                  # (optional, not in the repo) capsule_detector.tflite
├── lib/
│   └── main.dart            # Camera UI, controls, filters, detection, training, image stamping
├── linux/ macos/ web/ windows/   # Flutter platform scaffolding
├── codemagic.yaml           # CI configuration
├── icon.png                 # App icon source image
├── pubspec.yaml             # Dependencies, version, Flutter configuration
├── pubspec.lock             # Locked dependency versions
├── analysis_options.yaml    # Static-analysis rules
└── LICENSE                  # MIT license
```

Most of the application, including the detection algorithms, currently lives in `lib/main.dart`.

## Technology stack

| Package                                                                     | Role                                                    |
|-----------------------------------------------------------------------------|---------------------------------------------------------|
| [Flutter](https://flutter.dev/) / [Dart](https://dart.dev/)                 | UI framework and language                               |
| [`camera`](https://pub.dev/packages/camera)                                 | Preview, capture, focus, exposure, and zoom             |
| [`camera_android_camerax`](https://pub.dev/packages/camera_android_camerax) | CameraX implementation for Android                      |
| [`tflite_flutter`](https://pub.dev/packages/tflite_flutter)                 | Running the optional YOLOv8 TFLite model                |
| [`image_picker`](https://pub.dev/packages/image_picker)                     | Opening photos from the gallery                         |
| [`gal`](https://pub.dev/packages/gal)                                       | Saving images to the device gallery                     |
| [`path_provider`](https://pub.dev/packages/path_provider)                   | Temporary directory for the result file                 |
| [`shared_preferences`](https://pub.dev/packages/shared_preferences)         | Custom magnifications and the trained capsule profile   |
| [`flutter_launcher_icons`](https://pub.dev/packages/flutter_launcher_icons) | Generating launcher icons (dev)                         |
| [`flutter_lints`](https://pub.dev/packages/flutter_lints)                   | Lint rules (dev)                                        |
 
The contour algorithm uses only Dart and Flutter's own `dart:ui` and `compute` (isolates), so the interface stays responsive; it needs no external computer-vision library. Exact versions are in `pubspec.yaml` and `pubspec.lock`.

## Permissions and privacy

- **Camera:** required for the live preview and capture.
- **Photo library:** used to save results and to open photos from the gallery.
- Custom magnifications and the trained profile are stored locally with `shared_preferences`.
- Images are processed and saved on the device, and the neural network runs locally. The app does not upload images or sync accounts.
You can deny a permission; the related feature will then be unavailable.

## Troubleshooting

**The camera preview does not appear.** Check that the device has a working camera, grant camera permission in system settings, close other apps using the camera, and try a physical device instead of an emulator.

**Captures are not saved.** Check photo-library permission and free storage. On iOS, verify the photo-library usage descriptions in `Info.plist`.

**Too many or too few capsules are detected.** Adjust the sensitivity and the radius range in the Multi dialog, improve lighting and focus, run a multi-pass calculation, or fix the result with Edit mode and train the profile on the capsules you confirmed.

**Dust or fibers are counted as capsules.** Delete them in Edit mode; they are remembered and will not return on the next run. Lowering sensitivity also helps.

**Results got worse after training.** Training on a few unrepresentative capsules can skew the profile. Press **Reset** in the Multi dialog and train again on typical capsules of your sample.

**The AI switch does not appear.** The TFLite model was not loaded. Check that `assets/capsule_detector.tflite` exists, is registered under `flutter: assets:` in `pubspec.yaml`, and that `tflite_flutter` is installed. Without a model, the contour algorithm is used.

**Build fails with a missing `tflite_flutter` package.** Add `tflite_flutter` to `dependencies` in `pubspec.yaml` and run `flutter pub get`.

**Focus, exposure, or zoom behave differently.** These depend on camera hardware and the platform implementation; some devices expose narrower ranges or lack manual focus.

**iOS build fails on Windows.** iOS builds require macOS and Xcode. Use a Mac or a macOS CI service such as Codemagic.

**Dependency or SDK errors.**

```
flutter clean
flutter pub get
flutter doctor -v
```

## Roadmap

Ideas for future development, not promises:

- Real pixel-to-μm calibration per microscope, objective, and camera, with saved profiles.
- Ship a trained YOLOv8 model and document how to train and export one.
- Interface localization (English alongside Russian).
- Export of per-capsule measurements (CSV).
- Split `lib/main.dart` into separate modules and add tests for the detection algorithms.
- In-app gallery and history of captures.
- Screenshots and a short demo video.
- Wider device testing on Android and iOS.

## Contributing

Bug reports, suggestions, and pull requests are welcome.

1. Fork the repository and create a branch: `git checkout -b feature/my-improvement`.
2. Make your changes and run `dart format .`, `flutter analyze`, and `flutter test`.
3. Open a pull request describing the problem, the solution, and how you tested it.
For bug reports, include the device model, OS version, Flutter version, reproduction steps, and logs. Do not attach private specimen images unless you have permission to share them.

<div align="center">

## License

Distributed under the [MIT License](https://github.com/Miraelll/Microscope-LAB/blob/main/LICENSE).

---

**Microscope LAB** · Microscope photography and capsule counting with Flutter · [Report a bug](https://github.com/Miraelll/Microscope-LAB/issues)

 </div>