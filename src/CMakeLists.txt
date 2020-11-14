add_executable(tracker
    main.cpp
)
target_link_libraries(
    tracker ${OpenCV_LIBS}
)