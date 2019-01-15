using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using MPI;

namespace MPI_ImageFilter
{
    class Program
    { 
        static void Main(string[] args)
        {
            MPIController mpiController = new MPIController();

            using (new MPI.Environment(ref args))
            {
                if (Communicator.world.Rank == 0)
                {
                    //master process

                    mpiController.grayScaleMaster("../../../../data/pexels-photo-640x336.jpeg", "../../../data/gray_img640x336");
                    mpiController.grayScaleMaster("../../../../data/pexels-photo-1280x733.jpeg", "../../../data/gray_img1280x733");
                    mpiController.grayScaleMaster("../../../../data/animal-beagle-canine-2048x1174.jpg", "../../../data/gray_img2048x1174");

                    Console.WriteLine("\n");

                    mpiController.gaussianBlurMaster("../../../../data/pexels-photo-640x336.jpeg", "../../../data/blur_img640x336");
                    mpiController.gaussianBlurMaster("../../../../data/pexels-photo-1280x733.jpeg", "../../../data/blur_img1280x733");
                    mpiController.gaussianBlurMaster("../../../../data/animal-beagle-canine-2048x1174.jpg", "../../../data/blur_img2048x1174");
                }
                else
                {
                    //child process
                    
                    mpiController.grayScaleWorker();
                    mpiController.grayScaleWorker();
                    mpiController.grayScaleWorker();
                    
                    mpiController.gaussianBlurWorker();
                    mpiController.gaussianBlurWorker();
                    mpiController.gaussianBlurWorker();
                }
            }
        }
    }
}
