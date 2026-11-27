plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.pixelavatar.app"
    compileSdk {
        version = release(37) {
            minorApiLevel = 0
        }
    }

    defaultConfig {
        applicationId = "com.pixelavatar.app"
        minSdk = 28
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += listOf("arm64-v8a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        noCompress += listOf("tflite", "bin")
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2026.11.01"))
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.material3:material3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.1")

    // Google AI Edge LiteRT (native GPU + CPU acceleration for our 3 .tflite models)
    implementation("com.google.ai.edge.litert:litert:2.3.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")

    // Google AI Edge LiteRT-LM (100% on-device LLM fallback when AICore Gemini Nano is unavailable)
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.20.0")

    // Google ML Kit GenAI Prompt API (Pixel 10 built-in AICore Gemini Nano)
    implementation("com.google.mlkit:genai-prompt:1.0.0-beta5")

    // Google ML Kit GenAI Text-to-Speech API (Pixel 10 on-device 24 kHz diffusion voices via AICore)
    implementation("com.google.mlkit:genai-text-to-speech:1.0.0-alpha1")
}
