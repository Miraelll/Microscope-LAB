<div align="center">
  
# 🔬 Microscope LAB

### Microscope photography and microcapsule counting on your phone: manual image controls, optical filters, automatic circle detection, diameters in μm, and a stamped, scaled result saved to your gallery.

[![Flutter](https://img.shields.io/badge/Flutter-cross--platform-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Dart](https://img.shields.io/badge/Dart-3.13.5%2B-0175C2?logo=dart&logoColor=white)](https://dart.dev/)
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS-333333)](#platform-support)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

## Contents

- [Overview](#overview)
- [What's new](#whats-new)
- [Features](#features)
- [How capsule counting works](#how-capsule-counting-works)
- [Scale ruler and diameters: read this first](#scale-ruler-and-diameters-read-this-first)
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

**MicroscopeCamera** is a Flutter application for digital microscopy. Put a phone camera at the eyepiece or camera adapter of a microscope (or open an existing photo from the gallery), tune the image with exact numeric controls, choose an optical filter and a magnification, and let the app find round objects such as microcapsules, count them, and estimate their diameters in micrometres (μm).

The result is saved as a PNG with the filter applied, every detected capsule outlined and numbered, a scale ruler, and an information stamp with the count and diameter statistics.

> **Note:** the app interface is currently in Russian.

> **Important:** diameters and the scale ruler are estimates derived from the selected magnification, not a calibrated measurement. See [Scale ruler and diameters: read this first](#scale-ruler-and-diameters-read-this-first).

## What's new

- **Automatic capsule detection and counting** with a circle Hough transform, running in a background isolate so the interface stays responsive.
- **Multi-pass calculation:** run the detection over several brightness and sharpness values and get the average count, with the most representative pass applied.
- **Diameters in μm** for every detected capsule, plus mean, minimum, and maximum in the stamp.
- **Photo mode:** open any photo from the gallery and analyze it with the same tools as the live camera.
- **Manual correction:** tap a capsule to remove it, tap an empty spot to add one.
- **Spinbox controls** for brightness and sharpness (step 0.1), plus one-tap **auto-tuning**.
- **Richer stamp** with capsule statistics, filter, settings, date, and a scale ruler.

## Features

### Live camera and photo mode

- Live preview from the rear camera (the first available camera is used if there is no rear one), portrait orientation.
- Zoom within the range reported by the device, up to 8×.
- Tap the preview to set focus and exposure; focus is then locked. Use the focus button to switch between locked and automatic focus.
- Open an existing photo from the gallery and work with it exactly like a live frame; a button returns you to the live camera.
- Reset button restores the default adjustments.

Available ranges depend on the phone's camera hardware and operating system.

### Image controls

- **Brightness:** spinbox from -10.0 to 10.0 in 0.1 steps. In live mode it also adjusts the camera exposure offset.
- **Sharpness (contrast):** spinbox from 0.0 to 10.0 in 0.1 steps.
- **Auto:** estimates brightness and sharpness from the central part of a sample frame.

Brightness and sharpness change how the image looks. They cannot recover detail the sensor did not capture.

### Optical filters

| Filter           | Purpose                                                   |
| ---------------- | --------------------------------------------------------- |
| Standard         | Natural-looking baseline image                            |
| High Detail (HD) | Increased contrast to emphasize visible edges             |
| Clean White      | Cooler, slightly blue-toned image                         |
| Achromat B/W     | Monochrome-style image                                    |
| Dark Field       | Inverted, high-contrast appearance for visual exploration |

Filters apply to the preview, to capsule detection, and to the saved image. They are visual presets and do not replace real laboratory illumination.

### Capsule counting

- **Count:** a single detection pass with the current brightness and sharpness.
- **Multi-pass:** enter several brightness and sharpness values separated by spaces (defaults: `-0.5 0.0 0.5 1.0` and `1.0 2.0 3.0`), or fill them around the current values (±0.5). The app reports the average count across all passes.
- **Detection settings:** minimum and maximum capsule radius in pixels, and edge sensitivity.
- **Overlay:** detected capsules are outlined and numbered on screen and in the saved image.
- **Manual edit:** turn on edit mode to remove false detections or add missed capsules by tapping.

### Magnification and labels

- Built-in magnifications: `4x`, `10x`, `20x`, `40x`, `100x`.
- Enter your own value (for example `60x`); the list is saved locally between sessions.
- Name the sample; the name appears in the stamp.

### Capture helpers

- Composition grid.
- `9:16` and `3:4` preview framing.
- Optional 3-second countdown before capture, to avoid shaking the setup.

## How capsule counting works

1. The frame is downscaled to a working width of 320 px for speed.
2. Brightness, sharpness, and the selected filter are applied to the pixels.
3. The image is smoothed and Sobel gradients are computed; an adaptive edge threshold depends on the sensitivity setting.
4. A **circle Hough transform** votes for circle centers over the configured radius range, then each candidate is checked for a closed contour (enough of 16 points around the circle must lie on an edge), which rejects straight lines.
5. Overlapping and nested circles are removed (non-maximum suppression).
6. In multi-pass mode, steps 2 to 5 repeat for every brightness and sharpness pair. The pass whose count is closest to the average is kept, and its brightness and sharpness are applied to the image.

Radius settings are in pixels of the 320 px working image. Results depend on lighting, focus, and how round and well-contrasted the objects are, so check the overlay and correct it manually when needed.

## Scale ruler and diameters: read this first

The ruler is always drawn as a segment equal to 20% of the image width. Its length in μm is chosen from the magnification label:

| Magnification | Ruler length |
| ------------- | ------------ |
| up to 5x      | 500 μm       |
| up to 15x     | 200 μm       |
| up to 30x     | 100 μm       |
| up to 60x     | 50 μm        |
| above 60x     | 20 μm        |

Capsule diameters in μm are computed from this same relation. That makes them **approximate values that depend entirely on how well this table matches your real optical setup** (objective, eyepiece or adapter, camera sensor, zoom). The app does not calibrate pixels to micrometres for a specific microscope. For quantitative work, check the ruler against a stage micrometer with your own equipment and treat the result as valid only for that setup.

## Platform support

| Platform                   | Status                                  | Notes                                                 |
| -------------------------- | --------------------------------------- | ----------------------------------------------------- |
| Android                    | Primary target                          | Android Studio and Android SDK required for building  |
| iOS                        | Primary target                          | macOS and Xcode required to build and sign            |
| Windows, macOS, Linux, Web | Flutter scaffolding present, untested   | Not the main focus of this project                    |

Camera access, photo-library permissions, and some camera controls behave differently between devices. Test on real hardware before distributing a release.


## Getting started

### Requirements

- [Flutter SDK](https://docs.flutter.dev/get-started/install) compatible with the Dart constraint in `pubspec.yaml` (`^3.13.5`).
- [Git](https://git-scm.com/).
- **Android:** Android Studio, Android SDK, and an emulator or device.
- **iOS:** macOS, Xcode with command-line tools, and a simulator or device.

```bash
flutter --version
flutter doctor
```

### Clone and install

```bash
git clone https://github.com/Miraelll/MicroscopeCamera.git
cd MicroscopeCamera
flutter pub get
```

### Check the project

```bash
flutter analyze
flutter test
```

### Run

```bash
flutter devices
flutter run                   # or: flutter run -d <device-id>
```

For iOS, open `ios/Runner.xcworkspace` in Xcode, configure signing, and run on a simulator or an iPhone.

## Building a release

**Android**

```bash
flutter build apk --release          # build/app/outputs/flutter-apk/app-release.apk
flutter build appbundle --release    # for Google Play
```

**iOS** (macOS and Xcode required)

```bash
flutter build ios --release
```

Before publishing, configure the application ID or bundle identifier, signing, version, icons, and store metadata. A `codemagic.yaml` is included for CI builds with Codemagic.

## How to use

1. **Launch the app** and allow camera access.
2. **Get an image.** Attach the phone camera to the microscope, or open a photo from the gallery.
3. **Name the sample** and **choose the magnification** in the top bar.
4. **Tune the image:** brightness and sharpness spinboxes (or **Auto**), filter, zoom, focus.
5. **Count capsules:** press **Count** for one pass, or **Multi** to set several brightness and sharpness values and get an average.
6. **Review the overlay.** Turn on edit mode to remove false detections or add missed capsules.
7. **Save.** Press the shutter button (with the optional 3-second timer in live mode); the result goes to your photo library.

For repeatable work, keep illumination, camera position, and optical settings the same between shots.

## The saved image

Each save produces a PNG named `capsules_<timestamp>.png` containing:

- the photo with the selected filter, brightness, and sharpness applied;
- every detected capsule outlined, numbered, and labeled with its diameter in μm;
- a dark information strip at the bottom with four lines: sample name and magnification; capsule count (and the multi-pass average) with mean diameter and the min to max range; filter, brightness, and sharpness; and the capture date and time;
- the scale ruler in the bottom-right corner of the strip.

## Project structure

```
MicroscopeCamera/
├── android/                 # Android application and native configuration
├── ios/                     # iOS application and Xcode configuration
├── lib/
│   └── main.dart            # Camera UI, controls, filters, capsule detection, image stamping
├── linux/ macos/ web/ windows/   # Flutter platform scaffolding
├── codemagic.yaml           # CI configuration
├── icon.png                 # App icon source image
├── pubspec.yaml             # Dependencies, version, Flutter configuration
├── pubspec.lock             # Locked dependency versions
├── analysis_options.yaml    # Static-analysis rules
└── LICENSE                  # MIT license
```

Most of the application, including the detection algorithm, currently lives in `lib/main.dart`.

## Technology stack

| Package                                                                     | Role                                                    |
| --------------------------------------------------------------------------- | ------------------------------------------------------- |
| [Flutter](https://flutter.dev/) / [Dart](https://dart.dev/)                 | UI framework and language                               |
| [`camera`](https://pub.dev/packages/camera)                                 | Preview, capture, focus, exposure, and zoom             |
| [`camera_android_camerax`](https://pub.dev/packages/camera_android_camerax) | CameraX implementation for Android                      |
| [`image_picker`](https://pub.dev/packages/image_picker)                     | Opening photos from the gallery                         |
| [`gal`](https://pub.dev/packages/gal)                                       | Saving images to the device gallery                     |
| [`path_provider`](https://pub.dev/packages/path_provider)                   | Temporary directory for the result file                 |
| [`shared_preferences`](https://pub.dev/packages/shared_preferences)         | Storing custom magnifications locally                   |
| [`flutter_launcher_icons`](https://pub.dev/packages/flutter_launcher_icons) | Generating launcher icons (dev)                         |
| [`flutter_lints`](https://pub.dev/packages/flutter_lints)                   | Lint rules (dev)                                        |

Image analysis uses only Dart and Flutter's own `dart:ui` and `compute` (isolates); there is no external computer-vision library. Exact versions are in `pubspec.yaml` and `pubspec.lock`.

## Permissions and privacy

- **Camera:** required for the live preview and capture.
- **Photo library:** used to save results and to open photos from the gallery.
- Custom magnifications are stored locally with `shared_preferences`.
- Images are processed and saved on the device. The app does not upload images or sync accounts.

You can deny a permission; the related feature will then be unavailable.

## Troubleshooting

**The camera preview does not appear.** Check that the device has a working camera, grant camera permission in system settings, close other apps using the camera, and try a physical device instead of an emulator.

**Captures are not saved.** Check photo-library permission and free storage. On iOS, verify the photo-library usage descriptions in `Info.plist`.

**Too many or too few capsules are detected.** Adjust the sensitivity and the minimum and maximum radius in the Multi dialog, improve lighting and focus, try another filter, or fix the result with manual edit mode.

**Focus, exposure, or zoom behave differently.** These depend on camera hardware and the platform implementation; some devices expose narrower ranges or lack manual focus.

**iOS build fails on Windows.** iOS builds require macOS and Xcode. Use a Mac or a macOS CI service such as Codemagic.

**Dependency or SDK errors.**

```bash
flutter clean
flutter pub get
flutter doctor -v
```

## Roadmap

Ideas for future development, not promises:

- [ ] Real pixel-to-μm calibration per microscope, objective, and camera, with saved profiles.
- [ ] Interface localization (English alongside Russian).
- [ ] Export of per-capsule measurements (CSV).
- [ ] Split `lib/main.dart` into separate modules and add tests for the detection algorithm.
- [ ] In-app gallery and history of captures.
- [ ] Screenshots and a short demo video.
- [ ] Wider device testing on Android and iOS.

## Contributing

Bug reports, suggestions, and pull requests are welcome.

1. Fork the repository and create a branch: `git checkout -b feature/my-improvement`.
2. Make your changes and run `dart format .`, `flutter analyze`, and `flutter test`.
3. Open a pull request describing the problem, the solution, and how you tested it.

For bug reports, include the device model, OS version, Flutter version, reproduction steps, and logs. Do not attach private specimen images unless you have permission to share them.

<div align="center">
  
## License

Distributed under the [MIT License](LICENSE).

---

**Microscope LAB** · Microscope photography and capsule counting with Flutter · [Report a bug](https://github.com/Miraelll/Microscope-LAB/issues)

</div>

