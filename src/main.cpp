#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int processImage(const String& img)
{
    String imgName;
    try
    {
        imgName = samples::findFile(img, true, true);
    }
    catch (const Exception& ex)
    {
        std::cout << "Image file not found\n";
        return -1;
    }

    Mat src = imread(imgName);
    if (src.empty())
    {
        std::cerr << "Image file invalid\n";
        return -1;
    }

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    medianBlur(gray, gray, 5);
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                 gray.rows/16,  // change this value to detect circles with different distances to each other
                 100, 30, 1, 50 // change the last two parameters
            // (min_radius & max_radius) to detect larger circles
    );
    for( size_t i = 0; i < circles.size(); i++ )
    {
        Vec3i c = circles[i];
        Point center = Point(c[0], c[1]);
        // circle center
        circle( src, center, 1, Scalar(0,100,100), 3, LINE_AA);
        // circle outline
        int radius = c[2];
        circle( src, center, radius, Scalar(255,0,255), 3, LINE_AA);
    }

    imshow("img", src);
    waitKey();
    return 0;
}

int processVideo(const String& video)
{
    return 0;
}

int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, 
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
        return processImage(parser.get<String>("img"));
    }
    if (parser.has("video"))
    {
        return processVideo(parser.get<String>("video"));
    }

    return 0;
}
