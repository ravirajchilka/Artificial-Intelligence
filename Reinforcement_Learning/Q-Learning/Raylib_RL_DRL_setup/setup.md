Use https://github.com/raysan5/raylib.git to clone the Raylib, this repo contains .sln and .lib file.
Check CmakeLists file to see where theses files are located.
Both Raylib and torch with cuda are compiled using MSVC, once you create an empty build folder use command cmake .. -G "Visual Studio 17 2022" -T host=x64 to build.  

