
#include "stereo_tracking/patch_matcher.h"
#include <stereo_tracking/patch_matcher.h>
#include "boost/filesystem.hpp"
#include "halconcpp/HalconCpp.h"
#include "metrics/macros.h"
#include "opencv2/core/core.hpp"
#include "opencv2/dnn/dnn.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
namespace drive {
namespace perception {
namespace stereo_tracking {
namespace dcpst = drive::common::perception::stereo_tracking;
#define DEBUG_PATCH
//#undef DEBUG_PATCH
void mat2HObject(const cv::Mat& image, HalconCpp::HObject& Hobj) {
    int hgt = image.rows;
    int wid = image.cols;
    //  CV_8UC3
    if (image.type() == CV_8UC3) {
        std::vector<cv::Mat> imgchannel;
        cv::split(image, imgchannel);
        cv::Mat imgB = imgchannel[0];
        cv::Mat imgG = imgchannel[1];
        cv::Mat imgR = imgchannel[2];
        uchar* dataR = new uchar[hgt * wid];
        uchar* dataG = new uchar[hgt * wid];
        uchar* dataB = new uchar[hgt * wid];
        for (int i = 0; i < hgt; i++) {
            memcpy(dataR + wid * i, imgR.data + imgR.step * i, wid);
            memcpy(dataG + wid * i, imgG.data + imgG.step * i, wid);
            memcpy(dataB + wid * i, imgB.data + imgB.step * i, wid);
        }
        HalconCpp::GenImage3(&Hobj, "byte", wid, hgt, (Hlong)dataR, (Hlong)dataG, (Hlong)dataB);
        delete[] dataR;
        delete[] dataG;
        delete[] dataB;
        dataR = NULL;
        dataG = NULL;
        dataB = NULL;
    }  //  CV_8UCU1
    else if (image.type() == CV_8UC1) {
        uchar* data = new uchar[hgt * wid];
        for (int i = 0; i < hgt; i++) std::memcpy(data + wid * i, image.data + image.step * i, wid);
        HalconCpp::GenImage1(&Hobj, "byte", wid, hgt, (Hlong)data);
        delete[] data;
        data = NULL;
    }
}

PatchMatcher::PatchMatcher(const dcpst::PatchMatcherParam param) : _param(param) {}

void PatchMatcher::findMatch(const cv::Mat& left,
                             const cv::Mat& right,
                             const std::vector<cv::Rect2d>& vision_objects,
                             std::vector<cv::Rect2d>& masks,
                             std::vector<PointPairList>& matches) {
    cv::Mat left_gray, right_gray;

    if (left.type() == CV_8UC3) {
        cv::cvtColor(left, left_gray, CV_BGR2GRAY);
        cv::cvtColor(right, right_gray, CV_BGR2GRAY);
    } else {
        left_gray = left.clone();
        right_gray = right.clone();
    }
    if (_param.patch_matcher_method() == dcpst::PYRAMID_NCC) {
        findMatchByPyramidNcc(left_gray, right_gray, vision_objects, masks, matches);
    } else if (_param.patch_matcher_method() == dcpst::HALCON_NCC) {
        findMatchByHalconNcc(left_gray, right_gray, vision_objects, masks, matches);
    } else if (_param.patch_matcher_method() == dcpst::CUDA_NCC) {
        findMatchByNccWithCuda(left_gray, right_gray, vision_objects, masks, matches);
    }
    _matches = matches;
#ifdef DEBUG_PATCH
    cv::Mat output;
    drawMatch(output);
    std::string name = "/tmp/stereo_matcher/" + std::to_string(100000 + _frame) + ".png";
    cv::imwrite(name, output);
#endif
}

void PatchMatcher::findMatchByHalconNcc(const cv::Mat& left_gray,
                                        const cv::Mat& right_gray,
                                        const std::vector<cv::Rect2d>& vision_objects,
                                        std::vector<cv::Rect2d>& masks,
                                        std::vector<PointPairList>& matches) {
    HalconCpp::HImage hLeftImage, hRightImage;
    mat2HObject(left_gray, hLeftImage);
    mat2HObject(right_gray, hRightImage);
    matches.resize(vision_objects.size());
#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < vision_objects.size(); ++i) {
        PointPairList match;
        cv::Rect2d result;
        if (findMatchByHalconNcc(hLeftImage,
                                 hRightImage,
                                 vision_objects[i],
                                 masks[i],
                                 result,
                                 _param.threshold(),
                                 _param.pyramid_level())) {
            PointPair pair(vision_objects[i].x + vision_objects[i].width / 2.0,
                           vision_objects[i].y + vision_objects[i].height / 2.0,
                           result.x + result.width / 2.0,
                           result.y + result.height / 2.0);
            match.push_back(pair);
        }
        matches[i] = match;
    }
}

bool PatchMatcher::findMatchByHalconNcc(HalconCpp::HImage& hLeftImage,
                                        HalconCpp::HImage& hRightImage,
                                        const cv::Rect2d& box,
                                        cv::Rect2d& range,
                                        cv::Rect2d& result,
                                        double threshold,
                                        int pyramid_level) {
    double row1 = box.y;
    double col1 = box.x;
    double row2 = box.y + box.height;
    double col2 = box.x + box.width;
    try {
        // create ncc model
        HalconCpp::HObject region;
        HalconCpp::GenRectangle1(&region, row1, col1, row2, col2);
        HalconCpp::HObject templ;
        HalconCpp::ReduceDomain(hLeftImage, region, &templ);
        HalconCpp::HTuple modelId;
        HalconCpp::CreateNccModel(templ, "auto", 0.0, 0.0, "auto", "use_polarity", &modelId);
        // create search image.
        HalconCpp::HObject searchRegion;
        HalconCpp::GenRectangle1(
                &searchRegion, range.y, range.x, range.y + range.height, range.x + range.width);
        HalconCpp::HObject searchImage;
        HalconCpp::ReduceDomain(hRightImage, searchRegion, &searchImage);

        // begin to search
        HalconCpp::HTuple match_row, match_col, angle, score;
        HalconCpp::FindNccModel(searchImage,
                                modelId,
                                0,
                                0,
                                threshold,
                                1,
                                0.5,
                                "true",
                                pyramid_level,
                                &match_row,
                                &match_col,
                                &angle,
                                &score);
        if (score.Length() == 0) {
            return false;
        }
        double row_offset = double(match_row) - (row1 + row2) / 2;
        double col_offset = double(match_col) - (col1 + col2) / 2;
        //        if (fabs(row_offset) > 3) {  // sine the matching is epipolar, so the row offset
        //        should
        //            // not be large.
        //            return false;
        //        }
        result.x = box.x + col_offset;
        result.y = box.y + row_offset;
        result.width = box.width;
        result.height = box.height;

        HalconCpp::ClearNccModel(modelId);
        return true;
    } catch (HalconCpp::HException& e) {
        LOG(ERROR) << "find Ncc model halcon exception: " << e.ErrorCode() << ", "
                   << e.ErrorMessage();
    }
    return false;
}

void PatchMatcher::findMatchByPyramidNcc(const cv::Mat& left_gray,
                                         const cv::Mat& right_gray,
                                         const std::vector<cv::Rect2d>& vision_objects,
                                         std::vector<cv::Rect2d>& masks,
                                         std::vector<PointPairList>& matches) {
    SCOPED_LABELED_TIMER_LOG("findMatchByPyramidNcc");
    matches.resize(vision_objects.size());
#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < vision_objects.size(); ++i) {
        PointPairList match;
        cv::Rect2d result;
        //        if (vision_objects[i].height > 16 && vision_objects[i].width > 16) {
        if (findMatchByNcc(left_gray,
                           right_gray,
                           vision_objects[i],
                           masks[i],
                           result,
                           _param.threshold(),
                           _param.pyramid_level())) {
            PointPair pair(vision_objects[i].x + vision_objects[i].width / 2.0,
                           vision_objects[i].y + vision_objects[i].height / 2.0,
                           result.x + result.width / 2.0,
                           result.y + result.height / 2.0);
            match.push_back(pair);
        }
        //        }
        matches[i] = match;
    }
}

void findMatchPoint(cv::Mat& img, float& maxVal, float& secondVal, cv::Point& maxloc) {
    assert(img.depth() == CV_32FC1);
    maxVal = -0.5f, secondVal = -0.5f;
    for (int r = 0; r < img.rows; ++r) {
        float* cur = img.ptr<float>(r);
        float* pre = r == 0 ? NULL : img.ptr<float>(r - 1);
        float* nxt = r == img.rows - 1 ? NULL : img.ptr<float>(r + 1);
        for (int c = 0; c < img.cols; ++c) {
            bool is_local_maximum = true;
            if (pre != NULL && pre[c] > cur[c]) {
                is_local_maximum = false;
            }
            if (nxt != NULL && nxt[c] > cur[c]) {
                is_local_maximum = false;
            }
            if (c > 0 && cur[c] < cur[c - 1]) {
                is_local_maximum = false;
            }
            if (c < img.cols - 1 && cur[c] < cur[c + 1]) {
                is_local_maximum = false;
            }
            if (is_local_maximum) {
                float tmp = cur[c];
                if (maxVal < cur[c] && cur[c] > 0) {
                    tmp = maxVal;
                    maxVal = cur[c];
                    maxloc.x = c;
                    maxloc.y = r;
                }
                secondVal = tmp > secondVal ? tmp : secondVal;
            }
        }
    }
}

bool PatchMatcher::findMatchByNcc(const cv::Mat& leftImage,
                                  const cv::Mat& rightImage,
                                  const cv::Rect2d& box,
                                  cv::Rect2d& range,
                                  cv::Rect2d& result,
                                  double threshold,
                                  int pyramid_level) {
    SCOPED_LABELED_TIMER_LOG("findMatchByNcc");
    std::vector<cv::Mat> refs, tpls, results;
    cv::Mat templ = leftImage(box);
    cv::Mat search = rightImage(range);
    cv::buildPyramid(search, refs, pyramid_level);
    cv::buildPyramid(templ, tpls, pyramid_level);
    cv::Mat ref, tpl, res;
    for (int level = pyramid_level; level >= 0; level--) {
        ref = refs[level];
        tpl = tpls[level];
        res = cv::Mat::zeros(ref.size() + cv::Size(1, 1) - tpl.size(), CV_32FC1);
        if (level == pyramid_level) {
            cv::matchTemplate(ref, tpl, res, CV_TM_CCORR_NORMED);
        } else {
            cv::Mat mask;
            cv::pyrUp(results.back(), mask);
            cv::threshold(mask, mask, threshold, 1., CV_THRESH_TOZERO);
            cv::Mat mask8u;
            mask.convertTo(mask8u, CV_8U, 255);
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(mask8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

            for (size_t i = 0; i < contours.size(); ++i) {
                cv::Rect r = cv::boundingRect(contours[i]);
                r.width += 4;
                r.height += 4;
                r.x = cv::max(0, r.x - 2);
                r.y = cv::max(0, r.y - 2);
                cv::Rect cd = r + (tpl.size() - cv::Size(1, 1));
                if (cd.width + cd.x >= ref.cols) {
                    cd.x = std::max(0, ref.cols - cd.width);
                    cd.width = ref.cols - cd.x;
                }
                if (cd.height + cd.y >= ref.rows) {
                    cd.y = std::max(0, ref.rows - cd.height);
                    cd.height = ref.rows - cd.y;
                }
                if (r.width + r.x >= res.cols) {
                    r.x = std::max(0, res.cols - r.width);
                    r.width = res.cols - r.x;
                }
                if (r.height + r.y >= res.rows) {
                    r.y = std::max(0, res.rows - r.height);
                    r.height = res.rows - r.y;
                }
                cv::matchTemplate(ref(cd), tpl, res(r), CV_TM_CCORR_NORMED);
            }
        }
        results.push_back(res);
    }
    double minval, maxval;
    cv::Point minloc, maxloc;
    cv::minMaxLoc(res, &minval, &maxval, &minloc, &maxloc);
    float bestval, secondval;
    cv::Point best_point;
    findMatchPoint(res, bestval, secondval, best_point);
    CHECK(best_point.x == maxloc.x && best_point.y == maxloc.y)
            << "maxLoc (" << maxloc.x << "," << maxloc.y << "), findMatchPoint: (" << best_point.x
            << "," << best_point.y << "), best score " << bestval << ", second score " << secondval;
    if (!((secondval > 0 && bestval - secondval > 0.2) || bestval > 0.85)) {
        return false;
    }
    double x = maxloc.x;
    double y = maxloc.y;
    {
        cv::Point2d subpix = subpixelLocation(res, maxloc);
        x = subpix.x;
        y = subpix.y;
    }
    result.x = x + range.x;
    result.y = y + range.y;
    result.width = tpl.cols;
    result.height = tpl.rows;
    return true;
}

void PatchMatcher::findMatchByNccWithCuda(const cv::Mat& left_gray,
                                          const cv::Mat& right_gray,
                                          const std::vector<cv::Rect2d>& vision_objects,
                                          std::vector<cv::Rect2d>& masks,
                                          std::vector<PointPairList>& matches) {
    cv::cuda::GpuMat left_image(left_gray);
    cv::cuda::GpuMat right_image(right_gray);
    matches.resize(vision_objects.size());
#pragma omp parallel for num_threads(4)
    for (size_t i = 0; i < vision_objects.size(); ++i) {
        PointPairList match;
        cv::Rect2d result;
        if (vision_objects[i].height > 1 && vision_objects[i].width > 1) {
            if (findMatchByNccWithCuda(left_image,
                                       right_image,
                                       vision_objects[i],
                                       masks[i],
                                       result,
                                       0.9,
                                       _param.pyramid_level())) {
                PointPair pair(vision_objects[i].x + vision_objects[i].width / 2.0,
                               vision_objects[i].y + vision_objects[i].height / 2.0,
                               result.x + result.width / 2.0,
                               result.y + result.height / 2.0);
                match.push_back(pair);
            }
        }
        //        matches.push_back(match);
        matches[i] = match;
    }
}
bool PatchMatcher::findMatchByNccWithCuda(cv::cuda::GpuMat& leftImage,
                                          cv::cuda::GpuMat& rightImage,
                                          const cv::Rect2d& box,
                                          cv::Rect2d& range,
                                          cv::Rect2d& result,
                                          double threshold,
                                          int pyramid_level) {
    cv::cuda::GpuMat templ(leftImage, box);
    cv::cuda::GpuMat search(rightImage, range);
    cv::cuda::GpuMat res;
    _matching_ptr->match(search, templ, res);
    cv::Point min_loc, max_loc;
    double min_val, max_val;
    cv::cuda::minMaxLoc(res, &min_val, &max_val, &min_loc, &max_loc);
    if (max_val < threshold) {
        return false;
    }
    result.x = range.x + max_loc.x;
    result.y = range.y + max_loc.y;
    result.width = box.width;
    result.height = box.height;
    return true;
}

bool PatchMatcher::drawMatch(cv::Mat& output) {
    cv::Mat concat;
    cv::vconcat(_left, _right, concat);
    if (_left.type() == CV_8UC3) {
        output = concat;
    } else if (_left.type() == CV_8UC1) {
        output = cv::Mat(concat.size(), CV_8UC3);
        cv::cvtColor(concat, output, CV_GRAY2RGB);
    }
    int cols = _left.cols;
    int rows = _left.rows;
    for (size_t i = 0; i < _bboxes.size(); ++i) {
        cv::Scalar color = getRandomColor();
        cv::rectangle(output, _bboxes[i], color);
        if (_masks[i].x < 0) {
            _masks[i].width += _masks[i].x;
            _masks[i].x = 0;
        }
        cv::rectangle(
                output,
                cv::Rect2d(_masks[i].x, _masks[i].y + rows, _masks[i].width, _masks[i].height),
                color);
        // draw match points for this bbox
        for (size_t j = 0; j < _matches[i].size(); ++j) {
            auto& pp = _matches[i][j];
            cv::circle(output, cv::Point2d(pp.x1, pp.y1), 2, color, 1);
            cv::circle(output, cv::Point2d(pp.x2, pp.y2 + rows - 1), 2, color, 1);
            cv::line(output,
                     cv::Point2d(pp.x1, pp.y1),
                     cv::Point2d(pp.x2, pp.y2 + rows - 1),
                     color,
                     2);
        }
        //        std::string text = std::to_string(_disp[i]);
        //        cv::putText(output, text, cv::Point2d(_bboxes[i].x, _bboxes[i].y - 5),
        //                    cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 255, 255));
    }
    return true;
}

void PatchMatcher::evaluate(const cv::Mat& left,
                            const cv::Mat& right,
                            const std::vector<cv::Rect2d>& bboxes,
                            std::vector<cv::Rect2d>& ranges,
                            double& accuracy,
                            double& time_cost,
                            double& precision) {
    cv::Mat left_gray, right_gray;

    if (left.type() == CV_8UC3) {
        cv::cvtColor(left, left_gray, CV_BGR2GRAY);
        cv::cvtColor(right, right_gray, CV_BGR2GRAY);
    } else {
        left_gray = left.clone();
        right_gray = right.clone();
    }
    std::vector<PointPairList> halcon_matches;
    std::vector<PointPairList> pyramid_matches;

    std::chrono::high_resolution_clock::time_point t;
    double halcon_time = 0;
    double pyramid_time = 0;

    t = std::chrono::high_resolution_clock::now();
    findMatchByHalconNcc(left_gray, right_gray, bboxes, ranges, halcon_matches);
    halcon_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                          std::chrono::high_resolution_clock::now() - t)
                          .count();

    t = std::chrono::high_resolution_clock::now();
    findMatchByPyramidNcc(left_gray, right_gray, bboxes, ranges, pyramid_matches);
    pyramid_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                           std::chrono::high_resolution_clock::now() - t)
                           .count();

    LOG(ERROR) << "halcon time: " << halcon_time << " pyramid time " << pyramid_time;

    time_cost = pyramid_time - halcon_time;
    int total = 0;
    int same = 0;
    double total_offset = 0;
    for (size_t i = 0; i < bboxes.size(); ++i) {
        PointPairList& h_match = halcon_matches[i];
        PointPairList& p_match = pyramid_matches[i];
        if (h_match.size() == 0) continue;
        total += 1;
        if (p_match.size() == 0) continue;
        cv::Rect2d h_roi(h_match[0].x2 - bboxes[i].width / 2.0,
                         h_match[0].y2 - bboxes[i].height / 2.0,
                         bboxes[i].width,
                         bboxes[i].height);
        cv::Rect2d p_roi(p_match[0].x2 - bboxes[i].width / 2.0,
                         p_match[0].y2 - bboxes[i].height / 2.0,
                         bboxes[i].width,
                         bboxes[i].height);

        double intersection_area = (h_roi & p_roi).area();
        double union_area = (h_roi | p_roi).area();
        double iou = intersection_area / union_area;
        //        if (iou > 0.8) {
        //            same += 1;
        //            total_offset += std::abs(h_match[0].x2 - p_match[0].x2);
        //        }
        if (std::abs(h_match[0].x2 - p_match[0].x2) < 3) {
            same += 1;
            total_offset += std::abs(h_match[0].x2 - p_match[0].x2);
        }
    }
    if (total == 0)
        accuracy = 1.0;
    else
        accuracy = same * 1.0 / total;
    if (same == 0)
        precision = 0;
    else
        precision = total_offset * 1.0 / same;
}

cv::Point2d PatchMatcher::subpixelLocation(cv::Mat& img, cv::Point& location) {
    SCOPED_LABELED_TIMER_LOG("subpixelLocation");
    cv::Point2d result(location.x, location.y);
    assert(img.cols >= 3 && img.rows >= 3 && img.depth() == CV_32FC1);
    {
        // find end
        float t1 = img.at<float>(location.y, location.x - 1);
        float t2 = img.at<float>(location.y, location.x);
        float t3 = img.at<float>(location.y, location.x + 1);
        float x1 = location.x - 1;
        float x2 = location.x;
        float x3 = location.x + 1;
        result.x = location.x - 0.5 * (t3 - t1) / (t3 + t1 - 2 * t2);
    }
    {
        float t0 = img.at<float>(location.y - 1, location.x);
        float t1 = img.at<float>(location.y, location.x);
        float t2 = img.at<float>(location.y + 1, location.x);
        result.y = location.y - 0.5 * (t2 - t0) / (t2 + t0 - 2 * t1);
        if (fabs(location.y - result.y) > 0.75) {
            result.y = location.y;
        }
    }
    return result;
}
}  // namespace stereo_tracking
}  // namespace perception
}  // namespace drive
