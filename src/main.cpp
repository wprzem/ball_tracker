#include <opencv2/opencv.hpp>
#include <iostream>

int processImage(const cv::String& img)
{
    cv::String imgName = cv::samples::findFile(img, true, true);

    cv::Mat src = cv::imread(imgName);
    if (src.empty())
    {
        std::cerr << "Image file invalid\n";
        return -1;
    }

    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::medianBlur(gray, gray, 5);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 100, 200 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    for( size_t i = 0; i < circles.size(); i++ )
    {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
        // circle outline
        int radius = c[2];
        circle( src, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
    }

    cv::imshow("img", src);
    cv::waitKey();
    return 0;
}

int processVideo(const cv::String& videoName)
{
    cv::VideoCapture video(videoName);
    if (!video.isOpened())
    {
        std::cerr << "Video file invalid\n";
        return -1;
    }

    while(1)
    {
        cv::Mat frame;
        // Capture frame-by-frame
        video >> frame;

        // If the frame is empty, break immediately
        if (frame.empty())
            break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::medianBlur(gray, gray, 5);
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT, 1,
                    gray.rows/16,  // change this value to detect circles with different distances to each other
                    100, 30, 200, 300 // change the last two parameters
                // (min_radius & max_radius) to detect larger circles
        );
        for( size_t i = 0; i < circles.size(); i++ )
        {
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);
            // circle center
            circle( frame, center, 1, cv::Scalar(0,100,100), 3, cv::LINE_AA);
            // circle outline
            int radius = c[2]; 
            circle( frame, center, radius, cv::Scalar(255,0,255), 3, cv::LINE_AA);
        }
        // Display the resulting frame
        imshow( "Frame", frame );

        // Press  ESC on keyboard to exit
        char c = (char)cv::waitKey(25);
        if(c==27)
            break;
    }
    // When everything done, release the video capture object
    video.release();
    cv::destroyAllWindows();
    return 0;
}

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, 
        "{h help||don't cry}"
        "{v video||source video file}"
        "{i img||source img file}");

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    if (parser.has("img"))
    {
        return processImage(parser.get<cv::String>("img"));
    }
    if (parser.has("video"))
    {
        return processVideo(parser.get<cv::String>("video"));
    }

    return 0;
}
