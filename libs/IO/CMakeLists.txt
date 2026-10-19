# Macro to check pkg-config modules and produce a full-path library list
macro(pkg_check_modules_fullpath_libs PREFIX MODULE_NAME)
	# Call pkg_check_modules
	pkg_check_modules(${PREFIX} REQUIRED IMPORTED_TARGET ${MODULE_NAME})
	if(${PREFIX}_FOUND)
		set(${PREFIX}_FULLPATH_LIBRARIES "" PARENT_SCOPE)
		foreach(lib ${${PREFIX}_LIBRARIES})
			set(lib_path "")
			set(lib_prefix "")
			set(lib_ext "")
			if(WIN32)
				set(lib_ext ".lib")
			else()
				set(lib_prefix "lib")
				set(lib_ext ".a")
			endif()
			# Skip system libraries like 'm' on Windows
			if(WIN32 AND (lib STREQUAL "m"))
				continue()
			endif()
			# If lib already has a path, use as is
			if(EXISTS "${lib_prefix}${lib}${lib_ext}")
				set(lib_path "${lib_prefix}${lib}${lib_ext}")
			else()
				foreach(dir ${${PREFIX}_LIBRARY_DIRS})
					if(EXISTS "${dir}/${lib_prefix}${lib}${lib_ext}")
						set(lib_path "${dir}/${lib_prefix}${lib}${lib_ext}")
						break()
					endif()
				endforeach()
			endif()
			if(NOT lib_path STREQUAL "")
				list(APPEND ${PREFIX}_FULLPATH_LIBRARIES "${lib_path}")
			else()
				list(APPEND ${PREFIX}_FULLPATH_LIBRARIES "${lib}")
			endif()
		endforeach()
		set(${PREFIX}_FULLPATH_LIBRARIES "${${PREFIX}_FULLPATH_LIBRARIES}" PARENT_SCOPE)
		message(STATUS "Found ${MODULE_NAME}: ${${PREFIX}_VERSION} libs: ${${PREFIX}_FULLPATH_LIBRARIES}")
	else()
		set(${PREFIX}_FULLPATH_LIBRARIES "" PARENT_SCOPE)
		message(STATUS "${MODULE_NAME} not found, support will be disabled")
	endif()
endmacro()

# Find required packages
FIND_PACKAGE(PNG QUIET)
if(PNG_FOUND)
	INCLUDE_DIRECTORIES(${PNG_INCLUDE_DIRS})
	ADD_DEFINITIONS(${PNG_DEFINITIONS})
	SET(_USE_PNG TRUE CACHE INTERNAL "")
else()
	SET(PNG_LIBRARIES "")
endif()
FIND_PACKAGE(JPEG QUIET)
if(JPEG_FOUND)
	INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
	ADD_DEFINITIONS(${JPEG_DEFINITIONS})
	SET(_USE_JPG TRUE CACHE INTERNAL "")
else()
	SET(JPEG_LIBRARIES "")
endif()
# FIND_PACKAGE(PkgConfig REQUIRED)
# ---- libjxl 检测 ----
# 优先尝试手动路径（Windows MSVC 场景）
set(JPEGXL_ROOT "E:/HerryProj/install/libjxl-0.11.1" CACHE PATH "Root directory of libjxl installation")

if(WIN32 AND JPEGXL_ROOT)
    # 直接用用户指定路径，跳过 pkg-config
    find_path(JPEGXL_INCLUDE_DIR
        NAMES jxl/decode.h jxl/version.h
        PATHS "${JPEGXL_ROOT}/include"
        NO_DEFAULT_PATH
    )
    find_library(JPEGXL_LIB_JXL
        NAMES jxl
        PATHS "${JPEGXL_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    find_library(JPEGXL_LIB_CMS
        NAMES jxl_cms
        PATHS "${JPEGXL_ROOT}/lib"
        NO_DEFAULT_PATH
    )
    find_library(JPEGXL_LIB_THREADS
        NAMES jxl_threads
        PATHS "${JPEGXL_ROOT}/lib"
        NO_DEFAULT_PATH
    )

    if(JPEGXL_INCLUDE_DIR AND JPEGXL_LIB_JXL AND JPEGXL_LIB_CMS AND JPEGXL_LIB_THREADS)
        set(JPEGXL_FOUND TRUE)
        set(JPEGXL_LIBRARIES
            "${JPEGXL_LIB_JXL}"
            "${JPEGXL_LIB_CMS}"
            "${JPEGXL_LIB_THREADS}"
        )
        include_directories("${JPEGXL_INCLUDE_DIR}")
        SET(_USE_JXL TRUE CACHE INTERNAL "")
        message(STATUS "Found libjxl (manual): ${JPEGXL_LIBRARIES}")
    else()
        set(JPEGXL_FOUND FALSE)
        SET(JPEGXL_LIBRARIES "")
        message(STATUS "libjxl not found at JPEGXL_ROOT=${JPEGXL_ROOT}")
    endif()

else()
    # Linux / 没有指定 JPEGXL_ROOT 时走 pkg-config 原有逻辑
    pkg_check_modules_fullpath_libs(JPEGXL libjxl)
    if(JPEGXL_FOUND)
        SET(_USE_JXL TRUE CACHE INTERNAL "")
        set(JPEGXL_LIBRARIES ${JPEGXL_FULLPATH_LIBRARIES})
        include_directories(${JPEGXL_INCLUDE_DIRS})
    else()
        SET(JPEGXL_LIBRARIES "")
    endif()
endif()
# ---- libjxl 检测结束 ----
FIND_PACKAGE(TIFF QUIET)
if(TIFF_FOUND)
	INCLUDE_DIRECTORIES(${TIFF_INCLUDE_DIR})
	ADD_DEFINITIONS(${TIFF_DEFINITIONS})
	SET(_USE_TIFF TRUE CACHE INTERNAL "")
else()
	SET(TIFF_LIBRARIES "")
endif()

# List sources files
FILE(GLOB LIBRARY_FILES_C "*.cpp")
FILE(GLOB LIBRARY_FILES_H "*.h" "*.inl")

cxx_library_with_type(IO "Libs" "" "${cxx_default}"
	${LIBRARY_FILES_C} ${LIBRARY_FILES_H}
)

# Manually set Common.h as the precompiled header
IF(CMAKE_VERSION VERSION_GREATER_EQUAL 3.16.0)
	TARGET_PRECOMPILE_HEADERS(IO PRIVATE "Common.h")
endif()

# Link its dependencies
TARGET_LINK_LIBRARIES(IO PUBLIC Common ${PNG_LIBRARIES} ${JPEG_LIBRARIES} ${JPEGXL_LIBRARIES} ${TIFF_LIBRARIES} ${EXIV2_LIBS})

# Install
SET_TARGET_PROPERTIES(IO PROPERTIES
	PUBLIC_HEADER "${LIBRARY_FILES_H}")
INSTALL(TARGETS IO
	EXPORT OpenMVSTargets
	LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
	ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
	RUNTIME DESTINATION "${INSTALL_BIN_DIR}"
	PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/IO")
