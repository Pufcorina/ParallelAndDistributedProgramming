import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Console {
    public void run() {
        while(true){
            printMenu();
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            try {
                int input = Integer.parseInt(br.readLine());
                switch (input){
                    case 0: System.exit(0);
                    case 1: gaussianBlur(); break;
                    case 2: grayScale(); break;
                    default:
                        System.out.println("Invalid option");
                }
            } catch (IOException e) {
                System.out.println(e.getMessage());
            }
        }
    }

    private void grayScale() {
        Image img2048x1174 = new Image("../data/animal-beagle-canine-2048x1174.jpg");
        System.out.println("Time for 2048x1174 is: " + img2048x1174.exportGrayScaleImageToFile("./data/gray_img2048x1174") + " seconds");

        Image img640x336 = new Image("../data/pexels-photo-640x336.jpeg");
        System.out.println("Time for 640x336 is: " + img640x336.exportGrayScaleImageToFile("./data/gray_img640x336") + " seconds");

        Image img1280x733 = new Image("../data/pexels-photo-1280x733.jpeg");
        System.out.println("Time for 1280x733 is: " + img1280x733.exportGrayScaleImageToFile("./data/gray_img1280x733") + " seconds");
    }

    private void gaussianBlur() {
        int blurSize = getBlurSize();

        Image img2048x1174 = new Image("../data/animal-beagle-canine-2048x1174.jpg", blurSize);
        System.out.println("Time for 2048x1174 is: " + img2048x1174.exportGaussianBlurImageToFile("./data/blur_img2048x1174") + " seconds");

        Image img640x336 = new Image("../data/pexels-photo-640x336.jpeg", blurSize);
        System.out.println("Time for 640x336 is: " + img640x336.exportGaussianBlurImageToFile("./data/blur_img640x336") + " seconds");

        Image img1280x733 = new Image("../data/pexels-photo-1280x733.jpeg", blurSize);
        System.out.println("Time for 1280x733 is: " + img1280x733.exportGaussianBlurImageToFile("./data/blur_img1280x733") + " seconds");
    }

    private int getBlurSize() {
        System.out.println("Blur depth: ");
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        try {
            int input = Integer.parseInt(br.readLine());
            return input;
        } catch (IOException e) {
            System.out.println(e.getMessage());
            return 0;
        }
    }

    private void printMenu() {
        System.out.println();
        System.out.println("Choose which image filter do you want to apply:");
        System.out.println("1) Gaussian Blur");
        System.out.println("2) Gray scale");
        System.out.println("0) Exit");
        System.out.print("Your choice: ");
        System.out.println();
    }
}
