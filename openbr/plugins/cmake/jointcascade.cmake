set(BR_WITH_JOINTCASCADE OFF CACHE BOOL "Build JointCascade Face Detector")

if (${BR_WITH_JOINTCASCADE})
    find_package(JointCascade REQUIRED)
    set(BR_THIRDPARTY_LIBS ${BR_THIRDPARTY_LIBS} ${JointCascade_LIB})
else()
    set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/jointcascade.cpp)
endif()
