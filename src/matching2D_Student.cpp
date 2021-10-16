#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        // Reference: Udacity Lesson solution
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        //int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // Reference: Udacity Lesson solution
        // Solution by the past mentor: https://knowledge.udacity.com/questions/211123
        if (descSource.type() != CV_32F || descRef.type() != CV_32F)
        //if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
        cout << "FLANN matching";
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)
        // Reference: Udacity Lesson solution
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (NN) with n=" << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        // Reference: Udacity Lesson solution
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if ((*it)[0].distance < minDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if  (descriptorType.compare("BRIEF") == 0)
    {
        // Reference: https://docs.opencv.org/master/d1/d93/classcv_1_1xfeatures2d_1_1BriefDescriptorExtractor.html#ae3bc52666010fb137ab6f0d32de51f60
        int bytes = 32;
        bool use_orientation = false;

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(bytes, use_orientation);
    }

    else if  (descriptorType.compare("ORB") == 0)
    {
        // Reference: https://docs.opencv.org/master/db/d95/classcv_1_1ORB.html
        int 	nfeatures = 500;
        float 	scaleFactor = 1.2f;
        int 	nlevels = 8;
        int 	edgeThreshold = 31;
        int 	firstLevel = 0;
        int 	WTA_K = 2;
        auto 	scoreType = cv::ORB::HARRIS_SCORE;
        int 	patchSize = 31;
        int 	fastThreshold = 20; 

        extractor = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel,
                                    WTA_K, scoreType, patchSize, fastThreshold);
    }

    else if  (descriptorType.compare("FREAK") == 0)
    {
        // Reference: https://docs.opencv.org/master/df/db4/classcv_1_1xfeatures2d_1_1FREAK.html
        bool 	orientationNormalized = true;
        bool 	scaleNormalized = true;
        float 	patternScale = 22.0f;
        int 	nOctaves = 4;
        const std::vector<int> & selectedPairs = std::vector<int>();

        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized,
                                                    patternScale, nOctaves, selectedPairs);
    }
 
    else if  (descriptorType.compare("AKAZE") == 0)
    {
        // Reference: https://docs.opencv.org/master/d8/d30/classcv_1_1AKAZE.html
        auto 	descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        int 	descriptor_size = 0;
        int 	descriptor_channels = 3;
        float 	threshold = 0.001f;
        int 	nOctaves = 4;
        int 	nOctaveLayers = 4;
        auto 	diffusivity = cv::KAZE::DIFF_PM_G2;

        extractor = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                    threshold, nOctaves, nOctaveLayers, diffusivity);
    }
  
    else if  (descriptorType.compare("SIFT") == 0)
    {
        // Reference: https://docs.opencv.org/master/d7/d60/classcv_1_1SIFT.html
        /*
        int 	nfeatures = 0;
        int 	nOctaveLayers = 3;
        double 	contrastThreshold = 0.04;
        double 	edgeThreshold = 10;
        double 	sigma = 1.6;
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold,
                                                 edgeThreshold, sigma);
        */

        // Reference: Udacity Lesson solution
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;
    int minResponse = 100;
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // remove the unnecessary points on 4 border lanes to make calculation easier
    int count = 0;
    int neighber = 3; // parameter to neglect the lower value point in this range

    for(int r = neighber; r < dst_norm.rows-neighber; r++) {
        for(int c = neighber; c < dst_norm.cols-neighber; c++) {
            int response = (int)dst_norm.at<float>(r, c);
            if (response > minResponse) {
                int comp_cnt = 0;
                int comp_thresh = (2 * neighber +1) * (2 * neighber +1) -1;
                for (int i = -neighber; i <= neighber; i++) {
                    for (int j = -neighber; j <= neighber; j++) {
                        if (response > (int)dst_norm.at<float>(r+i,c+j)) {
                            comp_cnt += 1;
                        }
                    }
                }

                if (comp_cnt == comp_thresh) {
                    cv::KeyPoint newKeyPoint;
                    newKeyPoint.pt = cv::Point2f(c, r);
                    //newKeyPoint.pt = cv::Point2f(r, c);
                    newKeyPoint.size = 2 * apertureSize;
                    keypoints.push_back(newKeyPoint);
                    count += 1;
                }                  
                
            }
        }
    }
     
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;


    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{
    // Refer 2D Features Framework in OpenCV
    // Reference: https://docs.opencv.org/master/d9/d97/tutorial_table_of_content_features2d.html
    cv::Ptr<cv::FeatureDetector> detector;
    // (1)FAST
    if (detectorType.compare("FAST") == 0) {
        // Reference: Udacity Lesson solution
        int threshold = 30;  // difference between intensity of the central pixel and pixels of a circle around this pixel
        bool bNMS = true;    // perform non-maxima suppression on keypoints
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        detector = cv::FastFeatureDetector::create(threshold, bNMS, type); 
   }
    // (2)BRISK
    else if (detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();        
    }
    // (3)ORB
    else if (detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
    }
    // (4)AKAZE
    else if (detectorType.compare("AKAZE") == 0) {
        detector = cv::AKAZE::create();
    }
    // (5)SIFT  
    else if (detectorType.compare("SIFT") == 0) {
        //Reference: Udacity Lesson solution
        detector = cv::xfeatures2d::SIFT::create();
    }

    // Common
    double t = (double)cv::getTickCount();
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detector Results"; 
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}