python dataset_builder.py ^
More?   --metadata_csv "./FMA metadata/metadata.csv" ^
More?   --checksum "../fma_small/checksums" ^
More?   --audio_dir "../fma_small/" ^
More?   --output_dir "../tracks_data/" ^
More?   --duration 10 ^
More?   --log_scale

[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1948] error: dequantization failed!
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1908] error: dequantization failed!
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1908] error: dequantization failed!
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1880] error: part2_3_length (3360) too large for available bit count (3240)
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1880] error: part2_3_length (3328) too large for available bit count (3240)
Note: Illegal Audio-MPEG-Header 0x00000000 at offset 33361.
Note: Trying to resync...
Note: Skipped 1024 bytes in input.
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:wetwork():1389] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
D:\OneDrive - Brown University\Brown\Spring 2025\CSCI 1470\Final Project\CSCI1470-Melodic-Machines\dataset_builder.py:71: UserWarning: PySoundFile failed. Trying audioread instead.
  y, _ = librosa.load(
C:\Users\candi\anaconda3\envs\csci1470\Lib\site-packages\librosa\core\audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
        Deprecated as of librosa version 0.10.0.
        It will be removed in librosa version 1.0.
  y, sr_native = __audioread_load(path, offset, duration, dtype)
Note: Illegal Audio-MPEG-Header 0x00000000 at offset 22401.
Note: Trying to resync...
Note: Skipped 1024 bytes in input.
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:wetwork():1389] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\layer3.c:INT123_do_layer3():1908] error: dequantization failed!
Note: Illegal Audio-MPEG-Header 0x00000000 at offset 63168.
Note: Trying to resync...
Note: Skipped 1024 bytes in input.
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:wetwork():1389] error: Giving up resync after 1024 bytes - your stream is not nice... (maybe increasing resync limit could help).
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:do_readahead():1123] warning: Cannot read next header, a one-frame stream? Duh...
Error processing 099134 at ../fma_small/099/099134.mp3: 
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:do_readahead():1123] warning: Cannot read next header, a one-frame stream? Duh...
Error processing 108925 at ../fma_small/108/108925.mp3: 
[C:\vcpkg\buildtrees\mpg123\src\-66150af195.clean\src\libmpg123\parse.c:do_readahead():1123] warning: Cannot read next header, a one-frame stream? Duh...
Error processing 133297 at ../fma_small/133/133297.mp3: 

(csci1470) D:\OneDrive - Brown University\Brown\Spring 2025\CSCI 1470\Final Project\CSCI1470-Melodic-Machines>