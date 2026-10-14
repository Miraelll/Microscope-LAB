<div align="center">

# 🔬 MicroscopeCamera

### Turn your smartphone into a digital microscope camera.

A Flutter application for capturing microscope images, adjusting the live preview, and saving annotated specimen photographs on **Android and iOS**.

[![Flutter](https://img.shields.io/badge/Flutter-cross--platform-02569B?logo=flutter&logoColor=white)](https://flutter.dev/)
[![Dart](https://img.shields.io/badge/Dart-3.13.5%2B-0175C2?logo=dart&logoColor=white)](https://dart.dev/)
[![Platforms](https://img.shields.io/badge/platforms-Android%20%7C%20iOS-333333)](#platform-support)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Contents

- [Overview](#overview)
- [Features](#features)
- [Platform support](#platform-support)
- [Screenshots](#screenshots)
- [Getting started](#getting-started)
- [Running the application](#running-the-application)
- [Building a release](#building-a-release)
- [How to use](#how-to-use)
- [Image processing and annotations](#image-processing-and-annotations)
- [Project structure](#project-structure)
- [Technology stack](#technology-stack)
- [Permissions and privacy](#permissions-and-privacy)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Overview

**MicroscopeCamera** is a mobile microscope photography app built with Flutter. It provides a camera-focused workspace for observing specimens, adjusting the preview, and creating images with useful contextual annotations.

The app is designed for students, educators, hobbyists, and anyone who needs to document objects viewed through a microscope. A dark interface keeps attention on the image, while quick controls make it possible to tune the view and capture a specimen without leaving the camera screen.

The repository contains a shared Flutter codebase and platform folders for Android and iOS. The goal is to maintain one application experience across both mobile platforms.

> **Important:** The magnification value is a user-selected label, and the scale bar is currently generated from that selection. Neither should be treated as a calibrated measurement unless the complete microscope and camera setup has been independently calibrated.

## Features

### Live microscope view

- Live preview from the device camera.
- Prefers the rear camera when available.
- High-resolution camera capture using the Flutter camera plugin.
- Portrait-oriented camera interface.
- Tap the preview to set the focus and exposure point on supported devices.

### Image controls

- **Zoom:** adjust the camera zoom within the range reported by the device.
- **Brightness:** tune the preview and captured image.
- **Sharpness / contrast:** adjust the image appearance.
- **Automatic tuning:** estimate suitable brightness and sharpness from a sample frame.
- **Focus lock:** switch between automatic and locked focus where supported.
- **Reset:** restore the default image adjustments and standard filter.

Actual camera capabilities depend on the phone, camera hardware, and operating-system implementation.

### Laboratory-style filters

The app includes five preview and capture filters:

| Filter | Purpose |
|---|---|
| Standard | Natural-looking baseline image |
| High Detail (HD) | Increased contrast to emphasize visible edges |
| Clean White | Cooler, slightly blue-toned image |
| Achromat B/W | Monochrome-style image |
| Dark Field | Inverted/high-contrast appearance for visual exploration |

These are visual processing presets. They do not replace laboratory illumination techniques or scientific image analysis.

### Specimen labels and capture settings

- Add a specimen or substance name, such as `NaCl crystal` or `Blood sample`.
- Select a magnification label from the built-in list: `4x`, `10x`, `20x`, `40x`, or `100x`.
- Enter a custom magnification label; custom values are saved locally for later sessions.
- Toggle a composition grid.
- Switch the preview framing between `9:16` and `3:4`.
- Enable a 3-second capture countdown.

### Annotated photographs

Captured images are processed and saved to the device's photo library. The image annotation includes:

- Specimen name.
- Selected magnification label.
- Selected filter.
- Brightness and sharpness settings.
- Capture date and time.
- A visual scale bar.

The app creates a PNG image with a dark information strip along the bottom, making saved captures easier to identify and organize.

## Platform support

| Platform | Status | Notes |
|---|---|---|
| Android | Project target | Requires Android Studio / Android SDK for building |
| iOS | Project target | Requires macOS and Xcode to build and sign an iOS app |
| Windows, macOS, Linux, Web | Flutter project folders may be present | These platforms are not the primary target of this project |

The shared Flutter UI is intended for Android and iOS. Camera access, photo-library permissions, and some camera controls can behave differently between devices. Test on real hardware before distributing a release.

## Screenshots

Screenshots can be added here once representative captures from Android and iOS are available. For example:

```text
screenshots/
├── android-camera.png
├── android-controls.png
└── ios-camera.png
```

To display an image hosted in the repository, add it and use Markdown:

```md
![MicroscopeCamera live preview](screenshots/android-camera.png)
```

## Getting started

### Requirements

Install the following tools before building the project:

- [Flutter SDK](https://docs.flutter.dev/get-started/install) compatible with the Dart SDK constraint in `pubspec.yaml` (`^3.13.5`).
- Dart SDK, included with Flutter.
- [Git](https://git-scm.com/).
- **Android:** Android Studio, Android SDK, and an emulator or Android device.
- **iOS:** macOS, Xcode, Xcode command-line tools, and an iOS simulator or device.

Check your local environment:

```bash
flutter --version
flutter doctor
```

Resolve any required issues reported by `flutter doctor` before building.

### Clone the repository

```bash
git clone https://github.com/Miraelll/MicroscopeCamera.git
cd MicroscopeCamera
```

### Install dependencies

```bash
flutter pub get
```

### Check the project

```bash
flutter analyze
flutter test
```

The test command runs the tests currently present in the repository. Add device-specific tests as the app evolves.

## Running the application

List the available devices:

```bash
flutter devices
```

Run on a connected Android device or selected emulator:

```bash
flutter run
```

If more than one device is available, specify its ID:

```bash
flutter run -d <device-id>
```

For iOS, run the following commands on a Mac with Xcode configured:

```bash
open ios/Runner.xcworkspace
```

Select a simulator or connected iPhone in Xcode, configure signing if needed, and run the app. You can also use `flutter run -d <device-id>` after the iOS toolchain is configured.

## Building a release

### Android APK

```bash
flutter build apk --release
```

The APK is normally created at:

```text
build/app/outputs/flutter-apk/app-release.apk
```

For Google Play distribution, build an Android App Bundle:

```bash
flutter build appbundle --release
```

Before publishing, configure the application ID, app signing, version, icons, and store metadata.

### iOS

iOS release builds must be created on macOS with Xcode:

```bash
flutter build ios --release
```

For App Store distribution, configure the bundle identifier, Apple Developer signing team, certificates/profiles, app permissions, and App Store Connect metadata in Xcode. An archive can then be prepared for distribution through Xcode.

## How to use

1. **Launch the app.** Allow camera access when prompted.
2. **Position the specimen.** Place the phone camera at the microscope's camera/eyepiece adapter or use the device camera setup appropriate to your equipment.
3. **Set the specimen label.** Tap the name field in the top bar and enter a descriptive name.
4. **Choose a magnification label.** Select one of the presets or enter a custom value. This is a label, not an automatic measurement of optical magnification.
5. **Compose the view.** Toggle the grid and select the `9:16` or `3:4` framing.
6. **Adjust the image.** Set zoom, brightness, sharpness, filter, and focus to suit the specimen.
7. **Capture.** Optionally enable the 3-second timer, then tap the shutter button.
8. **Review the saved image.** The processed PNG is saved to the device photo library, subject to the relevant permission.

For repeatable scientific work, keep microscope illumination, camera position, and optical settings consistent. Calibrate the complete optical setup independently before using any scale reference for measurements.

## Image processing and annotations

The app applies its selected color filter to the captured image and adds a metadata strip containing the specimen label, magnification label, filter, adjustment values, and timestamp.

The displayed brightness and sharpness controls primarily affect the visual appearance of the image; they do not recover detail that was not captured by the camera sensor. The automatic adjustment feature analyzes a sample frame to estimate image settings and should be treated as a convenience rather than a scientific calibration procedure.

**Scale-bar caution:** the current scale-bar label is selected heuristically from the entered magnification label. It is not derived from a calibrated pixel-to-micrometre conversion for a specific microscope, objective, camera, or adapter. Do not use it for quantitative measurements until calibration is implemented and verified.

## Project structure

```text
MicroscopeCamera/
├── android/                 # Android application and native configuration
├── ios/                     # iOS application and Xcode configuration
├── lib/
│   └── main.dart            # Flutter app entry point and camera interface
├── linux/                   # Flutter desktop platform scaffolding
├── macos/                   # Flutter desktop platform scaffolding
├── web/                     # Flutter web platform scaffolding
├── windows/                 # Flutter desktop platform scaffolding
├── icon.png                 # App icon source image
├── pubspec.yaml             # Dependencies, version, and Flutter configuration
├── pubspec.lock             # Locked Dart/Flutter dependency versions
├── analysis_options.yaml    # Dart static-analysis configuration
└── LICENSE                  # Project license
```

The camera UI, controls, filter processing, and capture annotation logic currently live primarily in `lib/main.dart`.

## Technology stack

| Technology / package | Role |
|---|---|
| [Flutter](https://flutter.dev/) | Cross-platform UI framework |
| [Dart](https://dart.dev/) | Application programming language |
| [`camera`](https://pub.dev/packages/camera) | Camera preview, capture, focus, exposure, and zoom APIs |
| [`camera_android_camerax`](https://pub.dev/packages/camera_android_camerax) | CameraX implementation for Android |
| [`permission_handler`](https://pub.dev/packages/permission_handler) | Cross-platform permission handling dependency |
| [`path_provider`](https://pub.dev/packages/path_provider) | Access to temporary and application directories |
| [`gal`](https://pub.dev/packages/gal) | Saving images to the device gallery |
| [`shared_preferences`](https://pub.dev/packages/shared_preferences) | Persisting custom magnification labels locally |
| [`flutter_launcher_icons`](https://pub.dev/packages/flutter_launcher_icons) | Generating launcher icons |
| [`flutter_lints`](https://pub.dev/packages/flutter_lints) | Dart/Flutter lint rules |

Dependency versions are defined in `pubspec.yaml` and `pubspec.lock`.

## Permissions and privacy

MicroscopeCamera needs access to the camera to show a live preview and capture photographs. It may also request photo-library access to save captured images.

- Grant camera permission when prompted.
- Grant the relevant photo-library permission if you want to save captures.
- Permission prompts and settings differ between Android and iOS.
- Custom magnification labels are stored locally using `shared_preferences`.
- The current capture flow saves images on the device. This README does not imply cloud upload, account synchronization, or remote image processing.

Ensure the native Android and iOS permission descriptions and configuration match the app's actual behavior before publishing. Users should be able to deny permissions without being misled about why access is requested.

## Troubleshooting

### The camera preview does not appear

- Check that the device has a working camera.
- Grant camera permission in system settings.
- Close other apps that may be using the camera and restart the app.
- Try a physical device if the emulator does not provide a suitable camera.
- Run `flutter doctor` and inspect the debug output from `flutter run`.

### Captures are not saved

- Check photo-library permissions in the operating system settings.
- Confirm the device has available storage.
- Check the application logs for errors from the gallery-saving operation.
- On iOS, verify the photo-library usage descriptions and privacy configuration.

### Zoom, focus, or exposure controls behave differently

These features depend on the camera hardware and platform implementation. Some devices expose narrower ranges or do not support a requested operation. Test the behavior on the target phone.

### iOS build fails on Windows

A native iOS release build requires macOS and Xcode. On Windows, you can develop shared Dart/Flutter code and build Android, but you need a Mac or a macOS CI service to compile, sign, and distribute the iOS app.

### Dependency or SDK errors

```bash
flutter clean
flutter pub get
flutter doctor -v
```

If the problem persists, record the Flutter version, operating system, target device, and complete error output when opening an issue.

## Roadmap

Potential next steps for the project include:

- [ ] Add screenshots and a short demonstration video.
- [ ] Test and document behavior on a range of Android and iOS devices.
- [ ] Add a dedicated in-app gallery/history for microscope captures.
- [ ] Add export and sharing options for annotated images.
- [ ] Implement microscope-specific calibration for a physically accurate scale bar.
- [ ] Add repeatable capture profiles for different microscopes and objectives.
- [ ] Expand automated tests for image processing and capture workflows.
- [ ] Set up CI builds and release artifacts for Android and iOS.

Roadmap items are ideas for future development, not claims that these features are already implemented.

## Contributing

Contributions, bug reports, and feature suggestions are welcome.

1. Fork the repository.
2. Create a feature branch:

   ```bash
   git checkout -b feature/my-improvement
   ```

3. Make your changes and run the checks:

   ```bash
   dart format .
   flutter analyze
   flutter test
   ```

4. Commit your changes and push the branch.
5. Open a pull request describing the problem, the solution, and how it was tested.

For bug reports, include the device model, OS version, Flutter version, reproduction steps, and relevant logs. Do not attach private or sensitive specimen images unless you have permission to share them.

## License

This project is distributed under the [MIT License](LICENSE).

---

<div align="center">

**MicroscopeCamera** · Cross-platform microscope photography with Flutter

[Repository](https://github.com/Miraelll/MicroscopeCamera) · [Report a bug](https://github.com/Miraelll/MicroscopeCamera/issues)

</div>
