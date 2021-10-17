
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include <fstream>

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-50, bottom+50), cv::FONT_ITALIC, 0.5, currColor);
        //putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-50, bottom+20), cv::FONT_ITALIC, 0.5, currColor);
        //putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    // save image data
    //cv::imwrite("../result/test.png", topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // (1) Calculate the mean of the Lidar points for the robust calculation.
    // This calculation is needed only for either Prev or Curr because vehicle doesn't move so quickly.
    // (The Lidar points can be disperse, anyway )
    float xMean = 0;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1){
        xMean += it1->x; // world position in m with x facing forward from sensor
    }
    xMean = xMean/(float)lidarPointsPrev.size();

    cout << "xMean: " << xMean << endl;

    // (2) Robust calculation of the vehicle end position.
    // Calculate the average x distance of Lidar points near from the xMean.
    float distancePrev = 0;
    float distanceCurr = 0;

    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1){
        // world coordinates
        distancePrev += it1->x; // world position in m with x facing forward from sensor
    }
    distancePrev = distancePrev/(float)lidarPointsPrev.size();

    for (auto it1 = lidarPointsCurr.begin(); it1 != lidarPointsCurr.end(); ++it1){
        // world coordinates
        distanceCurr += it1->x; // world position in m with x facing forward from sensor
    }
    distanceCurr = distanceCurr/(float)lidarPointsCurr.size();

    cout << "distancePrev: " << distancePrev << endl;
    cout << "distanceCurr: " << distanceCurr << endl;

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //bool debug = false;
    bool debug = true;
    
    // Debug information
    int prevBBID, currBBID;
    int prevCount[prevFrame.boundingBoxes.size()][2];
    int currCount[currFrame.boundingBoxes.size()][2];
    int prevCountMax = 0;
    int currCountMax = 0;
    
    // Necessary information
    int bothBBID;
    int bothCount[currFrame.boundingBoxes.size()][2];
    int bothCountMax = 0;

    int i, j;

    // Initialize the matxices
    for (i = 0; i < prevFrame.boundingBoxes.size(); i++){
        prevCount[i][0] = -1;
        prevCount[i][1] = 0;
    }
    for (i = 0; i < currFrame.boundingBoxes.size(); i++){
        currCount[i][0] = -1;
        bothCount[i][0] = -1;
        currCount[i][1] = 0;
        bothCount[i][1] = 0;
    }

    // pixel coordinates 
    cv::Point ptMatchesPrev; 
    cv::Point ptMatchesCurr;
 
    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1){
        // Index of each matched keypoints
        int prev_idx = it1->queryIdx; 
        int curr_idx = it1->trainIdx;
      
        /*---------------- debug start-------------------*/
        // (1) Count matched keypoints in each prevFrame and currFrame.
        if (debug){
            i = 0;
            for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){
                ptMatchesPrev.x = prevFrame.keypoints[prev_idx].pt.x;
                ptMatchesPrev.y = prevFrame.keypoints[prev_idx].pt.y;         
                // check wether point is within the bounding box
                if (it2->roi.contains(ptMatchesPrev)){
                    prevCount[i][0] = it2->boxID;// Input bounding box id
                    prevCount[i][1] += 1;
                }
                if (prevCount[i][1] > prevCountMax){
                    prevCountMax = prevCount[i][1];
                    prevBBID = it2->boxID; // set max counts ID
                }
                i++;
            }

            i = 0;
            for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2){
                ptMatchesCurr.x =  currFrame.keypoints[curr_idx].pt.x;
                ptMatchesCurr.y =  currFrame.keypoints[curr_idx].pt.y;
                // check wether point is within the bounding box
                if (it2->roi.contains(ptMatchesCurr)){
                    currCount[i][0] = it2->boxID;// Input bounding box id
                    currCount[i][1] += 1;
                }
                if (currCount[i][1] > currCountMax){
                    currCountMax = currCount[i][1];
                    currBBID = it2->boxID; // set max counts ID
                }
                i++;
            }
        }
        /*---------------- debug end-------------------*/

        // (2) Count matched keypoints in both prevFrame and currFrame.
        i = 0;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){
            ptMatchesPrev.x = prevFrame.keypoints[prev_idx].pt.x;
            ptMatchesPrev.y = prevFrame.keypoints[prev_idx].pt.y;
            // check wether point is within the bounding box
            if (it2->roi.contains(ptMatchesPrev)){
                 for (auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); ++it3){
                     if (it3->boxID == it2->boxID){
                        ptMatchesCurr.x =  currFrame.keypoints[curr_idx].pt.x;
                        ptMatchesCurr.y =  currFrame.keypoints[curr_idx].pt.y;
                        if (it3->roi.contains(ptMatchesCurr)){
                            bothCount[i][0] = it3->boxID;// Input bounding box id
                            bothCount[i][1] += 1;
                        }
                        if (bothCount[i][1] > bothCountMax){
                            bothCountMax = bothCount[i][1];
                            bothBBID = it3->boxID;
                        }
                     }
                 }
            }
            i++;
        }        
    }

    // Return the bounding box ID with the highest number of matched keypoints
    bbBestMatches[0] = bothBBID;
    bbBestMatches[1] = bothBBID;
    //bbBestMatches[0] = prevBBID;
    //bbBestMatches[1] = currBBID;

    cout << "-----------------------------------------" << endl;
    cout << "bbBestMatches: both = " << bothBBID << endl;

    j = 0;
    for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2){
        cout << "bothCount[bbID]" << bothCount[j][0] << ", bothCount[count]" << bothCount[j][1] << endl;
        j++;  
    }   

    if (debug){
        cout << "-----------------------------------------" << endl;
        cout << "[debug]bbBestMatches: prev = " << prevBBID << endl;
        j = 0;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){
            cout << "prevCount[bbID]" << prevCount[j][0] << ", prevCount[count]" << prevCount[j][1] << endl;
            j++;  
        }
        cout << "-----------------------------------------" << endl;
        cout << "[debug]bbBestMatches: curr = " << currBBID << endl;
        j = 0;
        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2){
            cout << "currCount[bbID]" << currCount[j][0] << ", currCount[count]" << currCount[j][1] << endl;
            j++;    
        }
        cout << "-----------------------------------------" << endl;
    }
}
