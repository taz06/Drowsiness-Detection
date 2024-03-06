import cv2

# Mendapatkan informasi build OpenCV
build_info = cv2.getBuildInformation()

# Mencari bagian yang mencantumkan codec yang didukung
codec_info_start = build_info.find("Video I/O:") + len("Video I/O:")
codec_info_end = build_info.find("Other third-party libraries", codec_info_start)

# Mencetak informasi codec
codec_info = build_info[codec_info_start:codec_info_end].strip()
print("Supported Codecs:\n", codec_info)
