#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        std::cerr << "Usage: tts.exe <text-file>" << std::endl;
        return 1;
    }

    std::ifstream file(argv[1]);

    if (!file) {
        std::cerr << "Could not open text file." << std::endl;
        return 1;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    std::string text = buffer.str();

    if (text.empty()) {
        std::cerr << "Text file is empty." << std::endl;
        return 1;
    }

    // Replace characters that can interfere with PowerShell.
    for (char& c : text) {

        if (c == '\'') {
            c = ' ';
        }

        if (c == '\n' || c == '\r') {
            c = ' ';
        }
    }

    std::string command =
        "powershell.exe -NoProfile -Command "
        "\"Add-Type -AssemblyName System.Speech; "
        "$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$voice.Speak('"
        + text +
        "'); "
        "$voice.Dispose()\"";

    int result = std::system(command.c_str());

    if (result != 0) {
        std::cerr << "Speech command failed." << std::endl;
        return 1;
    }

    return 0;
}