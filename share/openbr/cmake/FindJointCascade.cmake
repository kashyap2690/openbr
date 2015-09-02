find_path(JointCascade_DIR include/regressor.h ${CMAKE_SOURCE_DIR}/3rdparty/*)

mark_as_advanced(JointCascade_DIR)
include_directories(${JointCascade_DIR})
include_directories(${JointCascade_DIR}/liblinear)
include_directories(${JointCascade_DIR}/liblinear/blas)

set(JointCascade_LIB ${JointCascade_DIR}/libs/libJointCascade.dylib)
