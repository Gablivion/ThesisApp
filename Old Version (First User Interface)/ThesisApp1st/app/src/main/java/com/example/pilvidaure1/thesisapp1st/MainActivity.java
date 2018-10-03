package com.example.pilvidaure1.thesisapp1st;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.os.CountDownTimer;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;
import android.telephony.SmsManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity implements CvCameraViewListener2
{
    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    public static final int JAVA_DETECTOR = 0;
    private Mat teplateR;
    private Mat teplateL;

    private Mat mZoomWindow;
    private Mat mZoomWindow2;
    private Mat mRgba;
    private Mat mGray;

    private File mCascadeFile;
    private File mCascadeFileEye;
    private File mCasadeFileEye_R;

    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEye;
    private CascadeClassifier mJavaDetectorEye_R;

    private int mDetectorType = JAVA_DETECTOR;
    private String[] mDetectorName;
    private float mRelativeFaceSize = 0.2f;
    private int mAbsoluteFaceSize = 0;
    private CameraBridgeViewBase mOpenCvCameraView;

    double xCenter = -1;
    double yCenter = -1;

    public double id_tl_rec_x = 0;
    public double id_tl_rec_y = 0;
    public double id_br_rec_x = 0;
    public double id_br_rec_y = 0;
    public double id_height = 0;
    public double id_width = 0;
    public double avg_tl_x = 0;
    public double avg_tl_y = 0;
    public double avg_br_x = 0;
    public double avg_br_y = 0;
    public double avg_width = 0;
    public double avg_height = 0;
    public int checker = 0;

    public double id_tl_rec_x1 = 0;
    public double id_tl_rec_y1 = 0;
    public double id_br_rec_x1 = 0;
    public double id_br_rec_y1 = 0;
    public double id_height1 = 0;
    public double id_width1 = 0;
    public double avg_tl_x1 = 0;
    public double avg_tl_y1 = 0;
    public double avg_br_x1 = 0;
    public double avg_br_y1 = 0;
    public double avg_width1 = 0;
    public double avg_height1 = 0;

    public double max;
    public double min;
    public String checkye;
    public String checkye2;
    public String answer_blink;
    public int shift_letter = 1;
    public int shift_letter2 = 1;

    public boolean blink_letter_tr = true;
    public boolean blink_letter_tm = true;
    public boolean blink_letter_tl = true;
    public boolean blink_letter_bm = true;
    public boolean blink_letter_bl = true;
    public boolean blink_letter_br = true;
    public boolean blink_del_char = true;

    public int blink_idenifier = 1;
    public boolean send_identifier = false;
    public String message_keep;
    public boolean blink_checker_f = true;

    private BaseLoaderCallback  mLoaderCallback = new BaseLoaderCallback(this)
    {
        @Override
        public void onManagerConnected(int status)
        {
            switch (status)
            {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try
                    {
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);

                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;

                        while ((bytesRead = is.read(buffer)) != -1)
                        {
                            os.write(buffer, 0, bytesRead);
                        }

                        is.close();
                        os.close();

                        InputStream ise = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);

                        mCascadeFileEye = new File(cascadeDirEye, "haarcascade_lefteye_2splits.xml");
                        FileOutputStream ose = new FileOutputStream(mCascadeFileEye);

                        while ((bytesRead = ise.read(buffer)) != -1)
                        {
                            ose.write(buffer, 0, bytesRead);
                        }

                        ise.close();
                        ose.close();

                        InputStream ise_r = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
                        File cascadeDirEye_R = getDir("cascade_r", Context.MODE_PRIVATE);

                        mCasadeFileEye_R = new File(cascadeDirEye_R, "haarcascade_righteye_2splits.xml");
                        FileOutputStream ose_r = new FileOutputStream(mCasadeFileEye_R);

                        while((bytesRead = ise_r.read(buffer)) != -1)
                        {
                            ose_r.write(buffer, 0, bytesRead);
                        }

                        ise_r.close();
                        ose_r.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        mJavaDetector.load( mCascadeFile.getAbsolutePath());

                        if (mJavaDetector.empty())
                        {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        }
                        else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mJavaDetectorEye = new CascadeClassifier(mCascadeFileEye.getAbsolutePath());
                        mJavaDetectorEye.load( mCascadeFileEye.getAbsolutePath());

                        if (mJavaDetectorEye.empty())
                        {
                            Log.e(TAG, "Failed to load cascade classifier for eye");
                            mJavaDetectorEye = null;
                        }
                        else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFileEye.getAbsolutePath());

                        mJavaDetectorEye_R = new CascadeClassifier(mCasadeFileEye_R.getAbsolutePath());
                        mJavaDetectorEye_R.load( mCasadeFileEye_R.getAbsolutePath());


                        if(mJavaDetectorEye_R.empty())
                        {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetectorEye_R  = null;
                        }
                        else
                            Log.i(TAG, "Loaded cascade classifier from " + mCasadeFileEye_R.getAbsolutePath());

                        cascadeDir.delete();
                        cascadeDirEye.delete();
                        cascadeDirEye_R.delete();
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableView();
                }
                break;

                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity()
    {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
    }

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();

        if (!OpenCVLoader.initDebug())
        {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        }
        else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy()
    {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height)
    {
        mGray = new Mat();
        mRgba = new Mat();
        mZoomWindow = new Mat();
        mZoomWindow2 = new Mat();
    }

    public void onCameraViewStopped()
    {
        mGray.release();
        mRgba.release();
        mZoomWindow.release();
        mZoomWindow2.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame)
    {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        if (mAbsoluteFaceSize == 0)
        {
            int height = mRgba.rows();
            if (Math.round(height * mRelativeFaceSize) > 0)
            {
                mAbsoluteFaceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect faces = new MatOfRect();

        //face
        if (mDetectorType == JAVA_DETECTOR) {
            if (mJavaDetector != null)
                mJavaDetector.detectMultiScale(mRgba, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                        new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());
        }
        else {
            Log.e(TAG, "Detection method is not selected!");
        }

        Rect[] facesArray = faces.toArray();

        for (int i = 0; i < facesArray.length; i++)
        {
            Imgproc.rectangle(mRgba, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 2);

            xCenter = (facesArray[i].x + facesArray[i].width + facesArray[i].x) / 2;
            yCenter = (facesArray[i].y + facesArray[i].y + facesArray[i].height) / 2;

            Point center = new Point(xCenter, yCenter);
            Imgproc.circle(mRgba, center, 10, new Scalar(255, 0, 0, 255), 2);
            Imgproc.putText(mRgba, "[" + center.x + "," + center.y + "]", new Point(center.x + 20, center.y + 20), Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(255, 255, 255, 255));

            Rect r = facesArray[i];

            Rect eyearea = new Rect(r.x + r.width / 8, (int) (r.y + (r.height / 4.5)), r.width - 2 * r.width / 8, (int) (r.height / 3.0));

            Rect eyearea_right = new Rect(r.x + r.width / 16, (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));
            Rect eyearea_left = new Rect(r.x + r.width / 16 + (r.width - 2 * r.width / 16) / 2, (int) (r.y + (r.height / 4.5)),
                    (r.width - 2 * r.width / 16) / 2, (int) (r.height / 3.0));

            Imgproc.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(), new Scalar(255, 0, 0, 255), 1);
            Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(), new Scalar(255, 0, 0, 255), 1);

            teplateR = get_template(mJavaDetectorEye, eyearea_right, 24);
            teplateL = get_template1(mJavaDetectorEye, eyearea_left, 24);
        }
        return mRgba;
    }

    private Mat get_template(CascadeClassifier clasificator, Rect area, int size)
    {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();

        Point iris = new Point();
        Rect eye_template = new Rect();

        clasificator.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());
        Rect[] eyesArray = eyes.toArray();

        if(blink_checker_f == true) {
            if (eyesArray.length == 0) {

                answer_blink = checkye;

                final Button btn_tl = (Button) findViewById(R.id.btn_tl);
                final Button btn_tm = (Button) findViewById(R.id.btn_tm);
                final Button btn_tr = (Button) findViewById(R.id.btn_tr);
                final Button btn_bl = (Button) findViewById(R.id.btn_bl);
                final Button btn_bm = (Button) findViewById(R.id.btn_bm);
                final Button btn_br = (Button) findViewById(R.id.btn_br);

                if (answer_blink == "Top Left") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_tr == false) {
                                btn_tr.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }

                if (answer_blink == "Top Mid") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_tm == false) {
                                btn_tm.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }
                if (answer_blink == "Top Right") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_tl == false) {
                                btn_tl.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }

                if (answer_blink == "Bottom Left") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_br == false) {
                                btn_br.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }

                if (answer_blink == "Bottom Mid") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_bm == false) {
                                btn_bm.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }

                if (answer_blink == "Bottom Right") {
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            if (blink_letter_bl == false) {
                                btn_bl.performClick();
                            }

                            new CountDownTimer(3000, 1000) {
                                public void onTick(long millisUntilFinished) {
                                }

                                public void onFinish() {
                                    blink_letter_tl = false;
                                    blink_letter_tm = false;
                                    blink_letter_tr = false;
                                    blink_letter_bl = false;
                                    blink_letter_bm = false;
                                    blink_letter_br = false;
                                    blink_del_char = false;
                                    blink_checker_f = true;
                                }
                            }.start();
                        }
                    });
                }
            }
        }

        for (int i = 0; i < eyesArray.length; )
        {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;

            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), (int) e.width, (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);
            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, FACE_RECT_COLOR, 3);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            Point point_iris = new Point(iris.x, iris.y);
            Imgproc.circle(mRgba, point_iris, 10, FACE_RECT_COLOR);

            Point eyePoint = new Point(iris.x - area.x, iris.y - area.y);

            //cropped eye region
            eye_template = new Rect((int) iris.x - size / 2 - 45, (int) iris.y - size / 2 - 23, 110, 70);

            if (checker < 50){
                id_tl_rec_x = id_tl_rec_x + eye_template.tl().x;
                id_tl_rec_y = id_tl_rec_y + eye_template.tl().y;
                id_br_rec_x = id_br_rec_x + eye_template.br().x;
                id_br_rec_y = id_br_rec_y +  eye_template.br().y;

                id_width = id_width +  eye_template.width;
                id_height = id_height +  eye_template.height;
            }

            if (checker == 50){
                avg_tl_x = id_tl_rec_x / 25;
                avg_tl_y = id_tl_rec_y / 25;
                avg_br_x = id_br_rec_x / 25;
                avg_br_y = id_br_rec_y / 25;

                avg_width = id_width / 25;
                avg_height = id_height / 25;

                id_tl_rec_x = avg_tl_x;
                id_tl_rec_y = avg_tl_y;
                id_br_rec_x = avg_br_x;
                id_br_rec_y = avg_br_y;
                id_width = avg_width;
                id_height = avg_height;

                blink_letter_tl = false;
                blink_letter_tm = false;
                blink_letter_tr = false;
                blink_letter_bl = false;
                blink_letter_bm = false;
                blink_letter_br = false;
            }
            checker = checker + 1;

            Imgproc.rectangle(mRgba, new Point(id_tl_rec_x, id_tl_rec_y), new Point(id_br_rec_x, id_br_rec_y), new Scalar(255, 0, 0, 255), 2);

            //Point
            Point TL = new Point(id_tl_rec_x, id_tl_rec_y);
            Point TR = new Point(id_br_rec_x, id_tl_rec_y);
            Point BL = new Point(id_tl_rec_x, id_br_rec_y);
            Point BR = new Point(id_br_rec_x, id_br_rec_y);
            Point TM = new Point(id_tl_rec_x + (avg_width / 2) - 2, id_tl_rec_y);
            Point BM = new Point(id_tl_rec_x + (avg_width / 2) - 2, id_br_rec_y);

            //Navigation Identifier
            Imgproc.putText(mRgba, "TL",TL, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));
            Imgproc.putText(mRgba, "TR",TR, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));
            Imgproc.putText(mRgba, "BL",BL, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));
            Imgproc.putText(mRgba, "BR",BR, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));
            Imgproc.putText(mRgba, "TM",TM, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));
            Imgproc.putText(mRgba, "BM",BM, Core.FONT_HERSHEY_SIMPLEX, 0.5, new Scalar(255, 255, 255, 255));

            //Arrow
            Imgproc.arrowedLine(mRgba, point_iris, TL, new Scalar(255, 255, 255, 255));
            Imgproc.arrowedLine(mRgba, point_iris, TR, new Scalar(255, 255, 255, 255));
            Imgproc.arrowedLine(mRgba, point_iris, BL, new Scalar(255, 255, 255, 255));
            Imgproc.arrowedLine(mRgba, point_iris, BR, new Scalar(255, 255, 255, 255));
            Imgproc.arrowedLine(mRgba, point_iris, TM, new Scalar(255, 255, 255, 255));
            Imgproc.arrowedLine(mRgba, point_iris, BM, new Scalar(255, 255, 255, 255));

            //Top Left
            double y_distance_tl = Math.abs(id_tl_rec_y - iris.y) * Math.abs(id_tl_rec_y - iris.y);
            double x_distance_tl = Math.abs(id_tl_rec_x - iris.x) * Math.abs(id_tl_rec_x - iris.x);
            double xy_distance_tl = y_distance_tl + x_distance_tl;
            double xy_answer_tl = Math.sqrt(xy_distance_tl);
            double answer_tl = Math.round(xy_answer_tl) -33;

            //Top Right
            double y_distance_tr = Math.abs(id_tl_rec_y - iris.y) * Math.abs(id_tl_rec_y - iris.y);
            double x_distance_tr = Math.abs(id_br_rec_x - iris.x) * Math.abs(id_br_rec_x - iris.x);
            double xy_distance_tr = y_distance_tr + x_distance_tr;
            double xy_answer_tr = Math.sqrt(xy_distance_tr);
            double answer_tr = Math.round(xy_answer_tr) -30;

            //Bottom Left
            double y_distance_bl = Math.abs(id_br_rec_y - iris.y) * Math.abs(id_br_rec_y - iris.y);
            double x_distance_bl = Math.abs(id_tl_rec_x - iris.x) * Math.abs(id_tl_rec_x - iris.x);
            double xy_distance_bl = y_distance_bl + x_distance_bl;
            double xy_answer_bl = Math.sqrt(xy_distance_bl);
            double answer_bl = Math.round(xy_answer_bl) - 32;

            //Bottom Right
            double y_distance_br = Math.abs(id_br_rec_y - iris.y) * Math.abs(id_br_rec_y - iris.y);
            double x_distance_br = Math.abs(id_br_rec_x - iris.x) * Math.abs(id_br_rec_x - iris.x);
            double xy_distance_br = y_distance_br + x_distance_br;
            double xy_answer_br = Math.sqrt(xy_distance_br);
            double answer_br = Math.round(xy_answer_br) -32;

            //Top Mid
            double y_distance_tm = Math.abs(id_tl_rec_y - iris.y) * Math.abs(id_tl_rec_y - iris.y);
            double x_distance_tm = Math.abs(((id_tl_rec_x + (avg_width / 2) - 2) - iris.x) * Math.abs(((id_tl_rec_x + (avg_width / 2) - 2) - iris.x)));
            double xy_distance_tm = y_distance_tm + x_distance_tm;
            double xy_answer_tm = Math.sqrt(xy_distance_tm);
            double answer_tm = Math.round(xy_answer_tm);

            //Bottom Mid
            double y_distance_bm = Math.abs(id_br_rec_y - iris.y) * Math.abs(id_br_rec_y - iris.y);
            double x_distance_bm = Math.abs(((id_tl_rec_x + (avg_width / 2) - 2) - iris.x) * Math.abs(((id_tl_rec_x + (avg_width / 2) - 2) - iris.x)));
            double xy_distance_bm = y_distance_bm + x_distance_bm;
            double xy_answer_bm = Math.sqrt(xy_distance_bm);
            double answer_bm = Math.round(xy_answer_bm) -10.5;

            if(answer_tl > answer_tr) {
                max = answer_tl;
            }
            else {
                max = answer_tr;
            }
            if(answer_bl > answer_br)
            {
                if(answer_bl > max) {
                    max = answer_bl;
                }
            }
            else {
                if(answer_br > max) {
                    max = answer_br;
                }
            }
            if(answer_tm > answer_bm){
                if(answer_tm > max){
                    max = answer_tm;
                }
            }
            else{
                if(answer_bm > max){
                    max = answer_bm;
                }
            }

            if(answer_tl > answer_tr) {
                min = answer_tr;
            }
            else {
                min = answer_tl;
            }
            if(answer_bl < answer_br)
            {
                if(answer_bl < min) {
                    min = answer_bl;
                }
            }
            else {
                if(answer_br < min) {
                    min = answer_br;
                }
            }
            if(answer_tm < answer_bm){
                if(answer_tm < min){
                    min = answer_tm;
                }
            }
            else{
                if(answer_bm < min){
                    min = answer_bm;
                }
            }

            if(min == answer_tl){
                checkye = "Top Left";
            }

            if(min == answer_tr){
                checkye = "Top Right";
            }

            if(min == answer_bl){
                checkye = "Bottom Left";
            }

            if(min == answer_br){
                checkye = "Bottom Right";
            }

            if(min == answer_tm){
                checkye = "Top Mid";
            }

            if(min == answer_bm){
                checkye = "Bottom Mid";
            }

            String number1_check = "OneCheck";
            Log.e(number1_check, "TL: " + answer_tl + " TR: " + answer_tr + " BL: " + answer_bl + " BR: " + answer_br + " TM: " + answer_tm + " BM: " + answer_bm + " MAX: " + max + " MIN: " + min);

            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    private Mat get_template1(CascadeClassifier clasificator, Rect area, int size)
    {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();

        Point iris = new Point();
        Rect eye_template = new Rect();

        clasificator.detectMultiScale(mROI, eyes, 1.15, 2, Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE, new Size(30, 30), new Size());

        Rect[] eyesArray = eyes.toArray();

        if(eyesArray.length == 0) {
            runOnUiThread(new Runnable() {
                Button btn_del_char =(Button)findViewById(R.id.btn_del_char);
                @Override
                public void run() {
                    if(blink_del_char == false) {
                        btn_del_char.performClick();
                    }

                    new CountDownTimer(4000, 1000) {
                        public void onTick(long millisUntilFinished) {
                        }

                        public void onFinish() {
                            blink_letter_tl = false;
                            blink_letter_tm = false;
                            blink_letter_tr = false;
                            blink_letter_bl = false;
                            blink_letter_bm = false;
                            blink_letter_br = false;
                            blink_del_char = false;
                        }
                    }.start();
                }
            });
        }

        for (int i = 0; i < eyesArray.length; )
        {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;

            Rect eye_only_rectangle = new Rect((int) e.tl().x, (int) (e.tl().y + e.height * 0.4), (int) e.width, (int) (e.height * 0.6));

            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);

            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 1);

            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;
            Point point_iris = new Point(iris.x, iris.y);
            Imgproc.circle(mRgba, point_iris, 10, FACE_RECT_COLOR);

            Point eyePoint = new Point(iris.x - area.x, iris.y - area.y);

            eye_template = new Rect((int) iris.x - size / 2 - 25, (int) iris.y - size / 2 - 10, 60, 50);

            if (checker < 51){
                id_tl_rec_x1 = id_tl_rec_x1 + eye_template.tl().x;
                id_tl_rec_y1 = id_tl_rec_y1 + eye_template.tl().y;
                id_br_rec_x1 = id_br_rec_x1 + eye_template.br().x;
                id_br_rec_y1 = id_br_rec_y1 +  eye_template.br().y;

                id_width1 = id_width1 +  eye_template.width;
                id_height1 = id_height1 +  eye_template.height;
            }
            if (checker == 51){
                avg_tl_x1 = id_tl_rec_x1 / 25.5;
                avg_tl_y1 = id_tl_rec_y1 / 25.5;
                avg_br_x1 = id_br_rec_x1 / 25.5;
                avg_br_y1 = id_br_rec_y1 / 25.5;

                avg_width1 = id_width1 / 25.5;
                avg_height1 = id_height1 / 25.5;

                id_tl_rec_x1 = avg_tl_x1;
                id_tl_rec_y1 = avg_tl_y1;
                id_br_rec_x1 = avg_br_x1;
                id_br_rec_y1 = avg_br_y1;
                id_width1 = avg_width1;
                id_height1 = avg_height1;

                blink_del_char = false;
            }
            checker = checker + 1;

            Imgproc.rectangle(mRgba, new Point(id_tl_rec_x1, id_tl_rec_y1), new Point(id_br_rec_x1, id_br_rec_y1), new Scalar(255, 0, 0, 255), 2);

            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void click_btn_bm(View v){
        blink_checker_f = false;

        Button btn_tl = (Button) findViewById(R.id.btn_tl);
        Button btn_tm = (Button) findViewById(R.id.btn_tm);
        Button btn_tr = (Button) findViewById(R.id.btn_tr);
        Button btn_bl = (Button) findViewById(R.id.btn_bl);

        if(blink_idenifier == 1) {
            if (shift_letter == 7) {
                shift_letter = 0;
            }

            shift_letter = shift_letter + 1;

            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            if (shift_letter == 1) {
                btn_tl.setText("A");
                btn_tm.setText("B");
                btn_tr.setText("C");
                btn_bl.setText("D");
            } else if (shift_letter == 2) {
                btn_tl.setText("E");
                btn_tm.setText("F");
                btn_tr.setText("G");
                btn_bl.setText("H");
            } else if (shift_letter == 3) {
                btn_tl.setText("I");
                btn_tm.setText("J");
                btn_tr.setText("K");
                btn_bl.setText("L");
            } else if (shift_letter == 4) {
                btn_tl.setText("M");
                btn_tm.setText("N");
                btn_tr.setText("O");
                btn_bl.setText("P");
            } else if (shift_letter == 5) {
                btn_tl.setText("Q");
                btn_tm.setText("R");
                btn_tr.setText("S");
                btn_bl.setText("T");
            } else if (shift_letter == 6) {
                btn_tl.setText("U");
                btn_tm.setText("V");
                btn_tr.setText("W");
                btn_bl.setText("X");
            } else {
                btn_tl.setText("Y");
                btn_tm.setText("Z");
                btn_tr.setText("(space)");
                btn_bl.setText("?");
            }
        }

        if(blink_idenifier == 2) {
            if (shift_letter2 == 3) {
                shift_letter2 = 0;
            }
            shift_letter2 = shift_letter2 + 1;

            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            if (shift_letter2 == 1) {
                btn_tl.setText("09");
                btn_tm.setText("0");
                btn_tr.setText("9");
                btn_bl.setText("8");
            } else if (shift_letter2 == 2) {
                btn_tl.setText("7");
                btn_tm.setText("6");
                btn_tr.setText("5");
                btn_bl.setText("4");
            } else {
                btn_tl.setText("3");
                btn_tm.setText("2");
                btn_tr.setText("1");
                btn_bl.setText("BACK");
            }
        }
    }

    public void click_btn_tl(View v){
        blink_checker_f = false;

        if(blink_idenifier == 1) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tl = (Button) findViewById(R.id.btn_tl);
            String s_tl = (String) btn_tl.getText();
            tv_textmessage.append(s_tl);
        }

        if(blink_idenifier == 2) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tl = (Button) findViewById(R.id.btn_tl);
            String s_tl = (String) btn_tl.getText();
            tv_textmessage.append(s_tl);
        }
    }

    public void click_btn_tm(View v){
        blink_checker_f = false;

        if(blink_idenifier == 1) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tm = (Button) findViewById(R.id.btn_tm);
            String s_tl = (String) btn_tm.getText();
            tv_textmessage.append(s_tl);
        }

        if(blink_idenifier == 2) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tm = (Button) findViewById(R.id.btn_tm);
            String s_tl = (String) btn_tm.getText();
            tv_textmessage.append(s_tl);
        }
    }

    public void click_btn_bl(View v){
        blink_checker_f = false;

        if(blink_idenifier == 1) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;
            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_bl = (Button) findViewById(R.id.btn_bl);
            String s_tl = (String) btn_bl.getText();
            tv_textmessage.append(s_tl);
        }

        if(blink_idenifier == 2) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            TextView tv_title = (TextView) findViewById(R.id.tv_title);

            Button btn_bl = (Button) findViewById(R.id.btn_bl);
            Button btn_tl = (Button) findViewById(R.id.btn_tl);
            Button btn_tm = (Button) findViewById(R.id.btn_tm);
            Button btn_tr = (Button) findViewById(R.id.btn_tr);

            String s_tl = (String) btn_bl.getText();
            if(s_tl == "BACK"){
                tv_title.setText("Message:");
                String delete_num = (String)tv_textmessage.getText().toString(); ;
                delete_num = delete_num.substring(0, delete_num.length() - delete_num.length());
                tv_textmessage.setText(delete_num);
                tv_textmessage.setText(message_keep);

                btn_tl.setText("A");
                btn_tm.setText("B");
                btn_tr.setText("C");
                btn_bl.setText("D");
                blink_idenifier = 1;
                send_identifier = false;
                shift_letter = 1;
                shift_letter2 = 1;
            }else {
                tv_textmessage.append(s_tl);
            }
        }
    }

    public void click_btn_tr(View v){
        blink_checker_f = false;

        if(blink_idenifier == 1) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tr = (Button) findViewById(R.id.btn_tr);
            String s_tl = (String) btn_tr.getText();
            if (s_tl == "(space)") {
                tv_textmessage.append(" ");
                Toast.makeText(this, "Space", Toast.LENGTH_SHORT).show();
            } else {
                tv_textmessage.append(s_tl);
            }
        }

        if(blink_idenifier == 2) {
            blink_letter_tl = true;
            blink_letter_tm = true;
            blink_letter_tr = true;
            blink_letter_bl = true;
            blink_letter_bm = true;
            blink_letter_br = true;
            blink_del_char = true;

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            Button btn_tr = (Button) findViewById(R.id.btn_tr);
            String s_tl = (String) btn_tr.getText();
            tv_textmessage.append(s_tl);
        }
    }

    public void click_btn_br(View v){
        blink_checker_f = false;

        Button btn_tl = (Button) findViewById(R.id.btn_tl);
        Button btn_tm = (Button) findViewById(R.id.btn_tm);
        Button btn_tr = (Button) findViewById(R.id.btn_tr);
        Button btn_bl = (Button) findViewById(R.id.btn_bl);
        Button btn_br = (Button) findViewById(R.id.btn_br);

        blink_letter_tl = true;
        blink_letter_tm = true;
        blink_letter_tr = true;
        blink_letter_bl = true;
        blink_letter_bm = true;
        blink_letter_br = true;
        blink_del_char = true;

        if(send_identifier == false) {

            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            TextView tv_title = (TextView) findViewById(R.id.tv_title);

            tv_title.setText("Number:");
            message_keep = (String)tv_textmessage.getText().toString();
            String deleteall = message_keep;
            deleteall = deleteall.substring(0, message_keep.length() - message_keep.length());
            tv_textmessage.setText(deleteall);

            btn_tl.setText("09");
            btn_tm.setText("0");
            btn_tr.setText("9");
            btn_bl.setText("8");

            send_identifier = true;
            blink_idenifier = 2;
        }
        else{
            TextView tv_textmessage = (TextView) findViewById(R.id.tv_textmessage);
            String s_tl = (String)tv_textmessage.getText().toString();

            Toast.makeText(this, message_keep, Toast.LENGTH_SHORT).show();
            Toast.makeText(this, s_tl, Toast.LENGTH_SHORT).show();

            if(s_tl.length() == 11){
                SmsManager smsManager = SmsManager.getDefault();
                smsManager.sendTextMessage(s_tl, null, message_keep, null, null);

                TextView tv_title = (TextView) findViewById(R.id.tv_title);
                tv_title.setText("Message:");

                message_keep = "";
                String delete_text = (String) tv_textmessage.getText().toString();
                delete_text = delete_text.substring(0, delete_text.length() - delete_text.length());
                tv_textmessage.setText(delete_text);

                btn_tl.setText("A");
                btn_tm.setText("B");
                btn_tr.setText("C");
                btn_bl.setText("D");
                blink_idenifier = 1;
                send_identifier = false;
                shift_letter = 1;
                shift_letter2 = 1;

                Toast.makeText(this, "Message Sent", Toast.LENGTH_SHORT).show();
            }
        }
    }

    public void delete_letter(View v){
        blink_letter_tl = true;
        blink_letter_tm = true;
        blink_letter_tr = true;
        blink_letter_bl = true;
        blink_letter_bm = true;
        blink_letter_br = true;
        blink_del_char = true;

        TextView tv_textmessage = (TextView)findViewById(R.id.tv_textmessage);
        String s_tl = (String)tv_textmessage.getText().toString();
        if(s_tl.length() != 0) {
            s_tl = s_tl.substring(0, s_tl.length() - 1);
            tv_textmessage.setText(s_tl);
        }
    }
}