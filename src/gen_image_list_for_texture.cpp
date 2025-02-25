#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <Eigen/Eigen>

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <string>

#define TIME_SCALE 1000000000.0

struct Pose {
    std::string timestamp;
    Eigen::Quaterniond qwl;
    Eigen::Vector3d twl;
};

Eigen::Matrix3d Rlc_front, Rlc_back;
Eigen::Vector3d tlc_front, tlc_back;
double time_offset;

void generate_image_pose_list(std::vector<Pose>& ref_poses, std::vector<std::string> image_timestamps, std::string list_save_path, int type)
{
    std::ofstream file;
    file.open(list_save_path, std::ios::app);

    Eigen::Matrix3d Rlc;
    Eigen::Vector3d tlc;
    if (type == 0) { // front
        Rlc = Rlc_front;
        tlc = tlc_front;
    } else if (type == 1) { // back
        Rlc = Rlc_back;
        tlc = tlc_back;
    }

    int idx = 0;
    for (auto& timestamp : image_timestamps) {
        while (idx < ref_poses.size() && stod(ref_poses[idx].timestamp) / TIME_SCALE < stod(timestamp) / TIME_SCALE + time_offset)
            ++idx;
        
        if (idx == 0) {
            std::cout << "image of timestamp " << timestamp << " is ignored" << std::endl;
            continue;
        }

        if (idx >= ref_poses.size()) {
            std::cout << "image after timestamp " << timestamp << " are ignored" << std::endl;
            break;
        }

        double t1 = stod(ref_poses[idx-1].timestamp) / TIME_SCALE;
        double t2 = stod(ref_poses[idx].timestamp) / TIME_SCALE;
        double t = stod(timestamp) / TIME_SCALE + time_offset;

        double a1 = (t - t1) / (t2 - t1);
        double a2 = 1 - a1;

        Eigen::Vector4d q1, q2;
        q1(0) = ref_poses[idx-1].qwl.x(); q1(1) = ref_poses[idx-1].qwl.y(); q1(2) = ref_poses[idx-1].qwl.z(); q1(3) = ref_poses[idx-1].qwl.w();
        q2(0) = ref_poses[idx].qwl.x(); q2(1) = ref_poses[idx].qwl.y(); q2(2) = ref_poses[idx].qwl.z(); q2(3) = ref_poses[idx].qwl.w();

        Eigen::Vector4d q = a1*q1 + a2*q2;
        q = q / q.norm();
        Eigen::Vector3d twl = a1*ref_poses[idx-1].twl + a2*ref_poses[idx].twl;
        Eigen::Quaterniond qwl(q);

        Eigen::Matrix3d Rwl = qwl.toRotationMatrix();
        Eigen::Matrix3d Rwc = Rwl * Rlc;
        Eigen::Vector3d twc = Rwl * tlc + twl;
        Eigen::Quaterniond qwc(Rwc);

        file << timestamp << " "
             << qwc.x() << " " << qwc.y() << " " << qwc.z() << " " << qwc.w() << " "
             << twc.x() << " " << twc.y() << " " << twc.z() << std::endl;
    }

    file.close();

    return;
}


int main(int argc, char **argv) {

    // 0. parameters and pathes define
    std::string ros_bag_file_path = "/media/lym/1A10B49E0A4AC5C7/2025-01-17-00-17-00.bag";
    std::string front_image_topic = "/front_camera_image/compressed";
    std::string back_image_topic = "/back_camera_image/compressed";
    std::string front_image_save_folder = "/home/lym/res/6F_recon/front";
    std::string back_image_save_folder = "/home/lym/res/6F_recon/back";
    std::string fast_lio_pose_path = "/home/lym/res/6F_recon/metadata/lio_traj.txt";
    std::string front_image_pose_save_path = "/home/lym/res/6F_recon/metadata/front_pose.txt";
    std::string back_image_pose_save_path = "/home/lym/res/6F_recon/metadata/back_pose.txt";

    double K[4] = {305.85, 304.87, 572.00, 578.97};
    double D[5] = {0.082996, -0.027906, 0.007620, -0.001084, 0.0};
    time_offset = -0.6273233;  // lio timestamp = image timestamp + time_offset (s)
    int height = 1152;
    int width = 1152;

    Eigen::Matrix3d Rcl_front, Rcl_back;
    Eigen::Vector3d tcl_front, tcl_back;
    Rcl_front << 0.0244905, -0.351308, -0.93594,
                 0.00815397, -0.936119, 0.351588,
                 -0.999667, -0.0162422, -0.0200615;
    tcl_front << 0.157101, 0.150698, -0.155186;
    Rcl_back << 0.0753357, 0.362986, 0.928744,
                0.0575196, -0.931422, 0.359367,
                0.995498, 0.0263478, -0.0910481;
    tcl_back << -0.20808, -0.0547745, -0.0558749;
    
    Rlc_front = Rcl_front.transpose();
    tlc_front = -Rlc_front * tcl_front;
    Rlc_back = Rcl_back.transpose();
    tlc_back = -Rlc_back * tcl_back;
    
    // 1. extract front and back images and do the undistortion
    rosbag::Bag bag;
    bag.open(ros_bag_file_path, rosbag::bagmode::Read);
    std::vector<std::string> topics;
    topics.push_back(front_image_topic);
    topics.push_back(back_image_topic);
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = K[0];
    camera_matrix.at<double>(0, 2) = K[2];
    camera_matrix.at<double>(1, 1) = K[1];
    camera_matrix.at<double>(1, 2) = K[3];
    cv::Mat distortion_coef = cv::Mat::zeros(4, 1, CV_64F);
    distortion_coef.at<double>(0, 0) = D[0];
    distortion_coef.at<double>(1, 0) = D[1];
    distortion_coef.at<double>(2, 0) = D[2];
    distortion_coef.at<double>(3, 0) = D[3];
    cv::Size imageSize = cv::Size(width, height);
    cv::Mat map1, map2;
    cv::fisheye::initUndistortRectifyMap(camera_matrix, distortion_coef, cv::Matx33d::eye(), camera_matrix, imageSize, CV_16SC2, map1, map2);

    std::vector<std::string> front_image_timestamps, back_image_timestamps;

    for(rosbag::MessageInstance const m : view) {
        std::string topic = m.getTopic();

        sensor_msgs::CompressedImage::ConstPtr img = m.instantiate<sensor_msgs::CompressedImage>();
        // cv::Mat image = cv::imdecode(cv::Mat(img->data), cv::IMREAD_COLOR);
        // cv::remap(image, image, map1, map2, cv::INTER_LINEAR);

        std::string save_path;
        if (topic == front_image_topic) {
            save_path = front_image_save_folder + "/" + std::to_string(img->header.stamp.toNSec()) + ".png";
            front_image_timestamps.push_back(std::to_string(img->header.stamp.toNSec()));
        }
        else if (topic == back_image_topic) {
            save_path = back_image_save_folder + "/" + std::to_string(img->header.stamp.toNSec()) + ".png";
            back_image_timestamps.push_back(std::to_string(img->header.stamp.toNSec()));
        }
        else {
            std::cout << "Error in topics" << std::endl;
            exit(0);
        }

        // cv::imwrite(save_path, image);
    }

    bag.close();


    // 2. read the trajectory file of fast-lio and interpolate the pose of images
    std::sort(front_image_timestamps.begin(), front_image_timestamps.end(), [](std::string& a, std::string& b) {
        return a < b;
    });
    std::sort(back_image_timestamps.begin(), back_image_timestamps.end(), [](std::string& a, std::string& b) {
        return a < b;
    });

    std::vector<Pose> lidar_poses;
    std::ifstream lidar_file;
    lidar_file.open(fast_lio_pose_path);
    std::string lidar_line;
    while (getline(lidar_file, lidar_line)) {
        std::stringstream ss(lidar_line);

        Pose pose;
        ss >> pose.timestamp
           >> pose.qwl.x() >> pose.qwl.y() >> pose.qwl.z() >> pose.qwl.w()
           >> pose.twl.x() >> pose.twl.y() >> pose.twl.z();

        lidar_poses.push_back(pose);
    }

    generate_image_pose_list(lidar_poses, front_image_timestamps, front_image_pose_save_path, 0);
    generate_image_pose_list(lidar_poses, back_image_timestamps, back_image_pose_save_path, 1);

    return 0;
}