
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

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
    // debug view
    bool debView = false;
    //bool debView = true;

    // pixel coordinates 
    cv::Point ptMatchesPrev; 
    cv::Point ptMatchesCurr;

    // Distance mean to remove the outliers
    float distanceMean = 0;            
            
    // Count matched keypoints in each prevFrame and currFrame with the same index.
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end(); ++it1){
        // Index of each matched keypoints
        int prev_idx = it1->queryIdx; 
        int curr_idx = it1->trainIdx;

        // x, y position of each matched keypoints
        ptMatchesPrev.x = kptsPrev[prev_idx].pt.x;
        ptMatchesPrev.y = kptsPrev[prev_idx].pt.y;   
        ptMatchesCurr.x = kptsCurr[curr_idx].pt.x;
        ptMatchesCurr.y = kptsCurr[curr_idx].pt.y;

        if ((boundingBox.roi.contains(ptMatchesPrev)) && (boundingBox.roi.contains(ptMatchesCurr))){
            boundingBox.keypoints.push_back(kptsCurr[curr_idx]);
            boundingBox.kptMatches.push_back(*it1);
            //boundingBox.kptMatches.push_back(kptMatches[it1]); // error
            //cout << "prev_idx: " << prev_idx << ", curr_idx: " << curr_idx << endl;
            distanceMean += (float) it1->distance;
        }
    }
    distanceMean =  distanceMean / (float) boundingBox.kptMatches.size();

    if (debView){
        cout << "Prev keypoint size: " << kptMatches.size() << endl;  
        cout << "kptMatches size inside boundingBox (including outliers): " << boundingBox.kptMatches.size() << endl;    
        cout << endl << "distanceMean: " << distanceMean << endl;
        cout << "Before erase points: " << endl;
        for (auto it1 = boundingBox.kptMatches.begin(); it1 != boundingBox.kptMatches.end() - 1; ++it1)
        { 
            cout << "it1->trainIdx: " << it1->trainIdx << ", it1->queryIdx: " << it1->queryIdx << endl;
        }
    }

    // Erase kptMatches if distance is a lot more than mean.
    for (int i = boundingBox.kptMatches.size(); i > -1; i--){ 
    //for (int i = 0; i < boundingBox.kptMatches.size(); i++){ // When erase at vector, the indent change after that. Be careful.
        if (boundingBox.kptMatches[i].distance > (int)(distanceMean * 2.)){ // cut off more than 2 times as mean
        //if (boundingBox.kptMatches[i].distance > (int)(distanceMean + 10.)){ // 10 is too small
            if (debView){
                cout << "Delete kptMatches with distance: " << boundingBox.kptMatches[i].distance << endl;
            }
            boundingBox.kptMatches.erase(boundingBox.kptMatches.begin() + i);      
        }
    }
    if (debView){
        cout << "Threshold: " << (int)(distanceMean * 2.) << endl;
        cout << "kptMatches size inside boundingBox (without outliers): " << boundingBox.kptMatches.size() << endl;    
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // debug view
    bool debView = false;
    //bool debView = true;
    if (debView){
        cout << "---------computeTTCCamera function: start---------" << endl;
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
        { 
            cout << "it1->trainIdx: " << it1->trainIdx << ", it1->queryIdx: " << it1->queryIdx << endl;
        }
        cout << "kptMatches size: " << kptMatches.size() << endl;
        cout << "kptsPrev size: " << kptsPrev.size() << ", kptsCurr size: " << kptsCurr.size() << endl;
        cout << "frameRate: " << frameRate << endl;
    }
    // Reference: Udacity Sf, Lesson 3 "compute_ttc_camera.cpp"
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    int i = 0;
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        //cout << "it1->trainIdx: " << it1->trainIdx << ", it1->queryIdx: " << it1->queryIdx << endl;

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        int j = 0;

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            //double minDist = 5.0; // min. required distance
            double minDist = 100.0; // min. required distance

            //cout << "for #1" << endl;

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            //cout << "for #2" << endl;

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            //cout << "iteration 1: " << i << ", iteration 2: " << j << endl;
            //cout << "distCurr: " << distCurr << ", distPrev: " << distPrev << endl;

            //cout << "for #3" << endl;

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
                // Debug
                //cout << "distRatio: " << distRatio << endl;
            }
            j++;
        } // eof inner loop over all matched kpts
        i++;
    }     // eof outer loop over all matched kpts

    //cout << "distRatios.size: " << distRatios.size() << endl;

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        cout << "TTC = NAN" << endl;
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();

    double dT = 1 / frameRate;
    //TTC = -dT / (1 - meanDistRatio);

    // TODO: STUDENT TASK (replacement for meanDistRatio)
    vector<double> distRatiosSort;
    double temp;
    double medianDistRatio;

    copy(distRatios.begin(), distRatios.end(), back_inserter(distRatiosSort));

    for (int i = 0; i < distRatiosSort.size() - 1; i++){
        for (int j = 0; j < distRatiosSort.size() - i; j++){
            if(distRatiosSort[j] > distRatiosSort[j+1]){
                temp = distRatiosSort[j];
                distRatiosSort[j] = distRatiosSort[j+1];
                distRatiosSort[j+1] = temp;
            }
        }
    }

    if (debView){
            cout << "distRatiosSort.size: " << distRatiosSort.size() << endl;
    }

    if (distRatiosSort.size() % 2 == 0){
        medianDistRatio = (distRatiosSort[distRatiosSort.size()/2 -1]+distRatiosSort[distRatiosSort.size()/2])/2;
        //cout << distRatiosSort.size()/2 -1 << std::endl;
        //cout << medianDistRatio << std::endl;      
    }
    else {
        medianDistRatio = distRatiosSort[(distRatiosSort.size()-1)/2];
    }

    TTC = -dT / (1 - medianDistRatio);

    if (debView){
        cout << "Camera TTC: " << TTC << endl;
    }
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // debug view
    //bool debView = false;
    bool debView = true;

    double maxTTC = 50;

    // (1) Calculate the mean of the Lidar points for the robust calculation.
    // This calculation is needed only for either Prev or Curr because vehicle doesn't move so quickly.
    // (The Lidar points can be disperse, anyway )
    float xMean = 0;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1){
        xMean += it1->x; // world position in m with x facing forward from sensor
    }
    xMean = xMean/(float)lidarPointsPrev.size();

    if (debView){
        cout << "xMean: " << xMean << endl;
    }

    // (2) Robust calculation of the vehicle end position.
    // Calculate the average x distance of Lidar points near from the xMean.
    float distancePrev = 0;
    float distanceCurr = 0;
    float distanceThre = 2.0;//Calculate the mean of distance only by Lidar points within threshold.

    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1){
        // world coordinates
        if ((it1->x < (xMean + distanceThre)) && (it1->x > (xMean - distanceThre))){
            distancePrev += it1->x; // world position in m with x facing forward from sensor
        }
    }
    distancePrev = distancePrev/(float)lidarPointsPrev.size();

    for (auto it1 = lidarPointsCurr.begin(); it1 != lidarPointsCurr.end(); ++it1){
        // world coordinates
        if ((it1->x < (xMean + distanceThre)) && (it1->x > (xMean - distanceThre))){
            distanceCurr += it1->x; // world position in m with x facing forward from sensor
        }
    }
    distanceCurr = distanceCurr/(float)lidarPointsCurr.size();

    if (debView){
        cout << "distancePrev: " << distancePrev << endl;
        cout << "distanceCurr: " << distanceCurr << endl;
    }

    // (3) Calculate TTC
    double deltaT = 1/frameRate; // [Hz]-->[second]
    TTC = ( (double)distanceCurr * deltaT )/( (double)distancePrev - (double)distanceCurr);

    if ((TTC < 0) || (TTC > maxTTC)){
        TTC = maxTTC;
    }

    if (debView){
        cout << "Lidar TCC: " << TTC << endl;
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    // debug view
    bool debView = false;
    //bool debView = true;
    
    // variables
    int prevCount[prevFrame.boundingBoxes.size()][2]; //[0]: boxID, [1]: matched count
    int currCount[currFrame.boundingBoxes.size()][2]; //[0]: boxID, [1]: matched count
    int bothCount[prevFrame.boundingBoxes.size()][currFrame.boundingBoxes.size()]; //[0]: prev matched count, [1]: curr matched count

    int i, j;

    // initialize
     for (i = 0; i < prevFrame.boundingBoxes.size(); i++){
         prevCount[i][0] = -1;
         prevCount[i][1] = 0;
    }  

    for (j = 0; j < currFrame.boundingBoxes.size(); j++){
         currCount[j][0] = -1;
         currCount[j][1] = 0;
    }

    for (i = 0; i < prevFrame.boundingBoxes.size(); i++){
        for (j = 0; j < currFrame.boundingBoxes.size(); j++){
         bothCount[i][j] = 0;
        }
    }      

    // pixel coordinates 
    cv::Point ptMatchesPrev; 
    cv::Point ptMatchesCurr;

    // (1) Calculate matched number for each prevBB and currBB
    /*
    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1){
        // Index of each matched keypoints
        int prev_idx = it1->queryIdx; 
        int curr_idx = it1->trainIdx;

        // x, y position of each matched keypoints
        ptMatchesPrev.x = prevFrame.keypoints[prev_idx].pt.x;
        ptMatchesPrev.y = prevFrame.keypoints[prev_idx].pt.y;   
        ptMatchesCurr.x =  currFrame.keypoints[curr_idx].pt.x;
        ptMatchesCurr.y =  currFrame.keypoints[curr_idx].pt.y;

        // Count matched keypoints in each prevFrame and currFrame separetely.
        i = 0;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){     
            // check wether point is within the bounding box
            if (it2->roi.contains(ptMatchesPrev)){
                prevCount[i][0] = it2->boxID;// Input bounding box id
                prevCount[i][1] += 1;
            }
            i++;
        }
        j = 0;
        for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2){
            // check wether point is within the bounding box
            if (it2->roi.contains(ptMatchesCurr)){
                currCount[j][0] = it2->boxID;// Input bounding box id
                currCount[j][1] += 1;
            }
            j++;
        }        
    }
    */

    // (2) For each prevBB, find the currBB with the most matched point
    for (auto it1 = matches.begin(); it1 != matches.end(); ++it1){
        // Index of each matched keypoints
        int prev_idx = it1->queryIdx; 
        int curr_idx = it1->trainIdx;

        // x, y position of each matched keypoints
        ptMatchesPrev.x = prevFrame.keypoints[prev_idx].pt.x;
        ptMatchesPrev.y = prevFrame.keypoints[prev_idx].pt.y;   
        ptMatchesCurr.x =  currFrame.keypoints[curr_idx].pt.x;
        ptMatchesCurr.y =  currFrame.keypoints[curr_idx].pt.y;

        // Count matched keypoints in each prevFrame and currFrame with the same index.
        i = 0;
        for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){     
            // check wether point is within the bounding box
            if (it2->roi.contains(ptMatchesPrev)){
                prevCount[i][0] = it2->boxID; // Input bounding box id
                prevCount[i][1] += 1; // Count up
                j = 0;
                for (auto it3 = currFrame.boundingBoxes.begin(); it3 != currFrame.boundingBoxes.end(); ++it3){
                    // check wether point is within the bounding box
                    if (it3->roi.contains(ptMatchesCurr)){
                        currCount[j][0] = it3->boxID; // Input bounding box id
                        currCount[j][1] += 1; // Count up

                        bothCount[i][j] += 1; // Count up
                    }
                    j++;
                }   
            }
            i++;
        }     
    }

    for (i = 0; i < prevFrame.boundingBoxes.size(); i++){
        // Initialize the variables
        int bothCountMax = 0;
        int bothID = -1;

        // For each prev bounding box, calculate the most matched curr bounding box.
        for (j = 0; j < currFrame.boundingBoxes.size(); j++){
            if (bothCount[i][j] > bothCountMax){
                bothCountMax = bothCount[i][j];
                bothID = j;
            }
        }
        bbBestMatches.insert(std::pair<int, int>(i, bothID));
        if (debView){
        cout << "bbBestMatches: prev= " << i << ", curr= " << bothID << endl;
        }
    }      
  
    //---------------- debug start -------------------
    if (debView){
    cout << "-----------------------------------------" << endl;
    j = 0;
    for (auto it2 = prevFrame.boundingBoxes.begin(); it2 != prevFrame.boundingBoxes.end(); ++it2){
        cout << "prevCount[bbID]" << prevCount[j][0] << ", prevCount[count]" << prevCount[j][1] << endl;
        j++;  
    }
    cout << "-----------------------------------------" << endl;
    j = 0;
    for (auto it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2){
        cout << "currCount[bbID]" << currCount[j][0] << ", currCount[count]" << currCount[j][1] << endl;
        j++;    
    }
    cout << "-----------------------------------------" << endl;
    }
    //---------------- debug end -------------------
}
