if (NOT ${BR_WITH_DETECTION})
    set(BR_EXCLUDED_PLUGINS ${BR_EXCLUDED_PLUGINS} plugins/classification/jointcascade.cpp)
endif()
