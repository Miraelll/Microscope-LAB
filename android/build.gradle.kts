allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}

subprojects {
    project.evaluationDependsOn(":app")
}

subprojects {
    tasks.withType<JavaCompile>().configureEach {
        options.compilerArgs.add("-Xlint:-options")
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}

subprojects {
    val fixCompileSdk = Action<Project> {
        extensions.findByType(com.android.build.gradle.LibraryExtension::class.java)?.apply {
            compileSdk = 36
        }
    }
    if (state.executed) {
        fixCompileSdk.execute(this)
    } else {
        afterEvaluate(fixCompileSdk)
    }

    configurations.all {
        resolutionStrategy.eachDependency {
            if (requested.group == "org.tensorflow" &&
                (requested.name == "tensorflow-lite" ||
                 requested.name == "tensorflow-lite-gpu" ||
                 requested.name == "tensorflow-lite-api" ||
                 requested.name == "tensorflow-lite-gpu-api")) {
                useVersion("2.16.1")
            }
        }
    }

    if (project.name != "app") {
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
            }
        }
        tasks.withType<JavaCompile>().configureEach {
            sourceCompatibility = JavaVersion.VERSION_17.toString()
            targetCompatibility = JavaVersion.VERSION_17.toString()
        }
    }
}