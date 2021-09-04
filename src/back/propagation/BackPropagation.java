/*
 * Copyright (C) 2020 Miguel Angel Barrero Díaz
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package back.propagation;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.apache.commons.io.FileUtils.readLines;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.stat.StatUtils;
import static org.apache.commons.math3.util.FastMath.abs;

/**
 *
 * @author Miguel Angel Barrero Díaz
 */
public class BackPropagation {

    /**
     * @param args the command line arguments
     * @throws java.io.FileNotFoundException
     * @throws java.io.IOException
     */
    public static void main(String[] args) throws FileNotFoundException, IOException {
        System.out.println("The number of layers is: "+args[0]);
        System.out.println("The number of inpusts is: "+args[1]+"\n\n");
        
        //----------------------------------------------------------------------
        //Initialize variables
        //----------------------------------------------------------------------
        //m number of layers of the net
        int M = Integer.parseInt(args[0]);
        //number of inputs to the neural net,i.e J1 length
        int nx = Integer.parseInt(args[1]);
        //Vector to store the length of J for each layer m,i.e the output of the previous net
        int[] vectorJM  = new int[M-1];
       
        for(int i=0;i<vectorJM.length;i++){
            //Storing the numer of outputs in function of the number of neurons
            vectorJM[i]= Integer.parseInt(args[i+2]); 
        }
     
        //----------------------------------------------------------------------
        //Using cross validation to assest parameters
        //----------------------------------------------------------------------
        //If the prameter "cross" is included the program will use cross validation
        if("cross".equals(args[args.length-1])){
            File datafile = new File("Data/A1-turbine.txt");
            List<RealMatrix> fold1;
            List<RealMatrix> fold2;
            List<RealMatrix> fold3;
            //Create first fold
            fold1 = createFold(datafile,1);
            //Create second fold
            fold2 = createFold(datafile,2);
            //Create third fold
            fold3 = createFold(datafile,3);
            
            double error1 = 0;
            double error2 = 0;
            double error3 = 0;
            //---------------------------------
            //Train and predict with first fold
            //---------------------------------
            int[] config = new int[5];
            config[0] = vectorJM[M-2];
            config[1] = nx;
            config[2] = fold1.get(0).getRowDimension();
            config[3] = M;
            RealMatrix errorFold1;
            double min1 = StatUtils.min(fold1.get(1).getColumn(4));
            double max1 = StatUtils.max(fold1.get(1).getColumn(4));
            List<RealMatrix> weigthsList1 = createWeigths(vectorJM,nx,M);
            Netcalc.Mw = weigthsList1;
            
            for(int i=0;i<75000;i++){
                errorFold1 = Epoch.computeError(fold1.get(0),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,config);
                double num1 = StatUtils.sum(errorFold1.getColumn(0));
                double den1 = StatUtils.sum(errorFold1.getColumn(1)); 
                //double error1 = (StatUtils.sum(errorFold1.getColumn(2)))/config[2];
                error1 = 100*abs(num1)/abs(den1);
                //System.out.println(i);
                //System.out.println("The error after the training process is: "+ error1+"%");
                
                errorFold1 = Epoch.test(fold1.get(1),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,4,0,133,M,false);
                double[] output1 = errorFold1.getColumn(1);
                for(int j=0;j<output1.length;j++){
                    output1[j] = min1+(max1-min1)*output1[j];
                    //System.out.println(output1[i]);
                }
                num1 = StatUtils.sum(errorFold1.getColumn(0));
                den1 = StatUtils.sum(errorFold1.getColumn(1)); 
                error1 =100*num1/den1;
                //System.out.println("The validation error for the first fold is: "+error1+" %\n\n");
                if(i==75000-1){
                    System.out.println("The validation error for the first fold is: "+error1+" %\n\n");
                    break;
                }
                
                //Write to file
                writeToCsv("fold-1.csv",fold1.get(1),output1,min1,max1,error1);
            }
            
            //-----------------------------------
            //Train and validate with second fold
            //-----------------------------------
            
            //Add the two train chunks to the same data matrix
            for(int i=0;i<fold2.get(0).getRowDimension()-1;i++){
                fold2.get(0).setRowMatrix(fold2.get(0).getRowDimension()-1,fold2.get(2).getRowMatrix(i));
                }
            
            config[2] = fold2.get(0).getRowDimension();
            RealMatrix errorFold2;
            double min2 = StatUtils.min(fold2.get(1).getColumn(4));
            double max2 = StatUtils.max(fold2.get(1).getColumn(4));
            List<RealMatrix> weigthsList2 = createWeigths(vectorJM,nx,M);
            Netcalc.Mw = weigthsList2;
            
            for(int i=0;i<75000;i++){
                errorFold2 = Epoch.computeError(fold2.get(0),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,config);
                double num2 = StatUtils.sum(errorFold2.getColumn(0));
                double den2 = StatUtils.sum(errorFold2.getColumn(1)); 
                //double error2 = (StatUtils.sum(errorFold2.getColumn(2)))/config[2];
                error2 = 100*abs(num2)/abs(den2);
                //System.out.println(i);
                //System.out.println("The error after the training process is: "+ error2+"%");
               
                errorFold2 = Epoch.test(fold2.get(1),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,4,0,133,M,false);
                double[] output2 = errorFold2.getColumn(1);
                for(int j=0;j<output2.length;j++){
                    output2[j] = min2+(max2-min2)*output2[j];
                    //System.out.println(output2[i]);
                }
                num2 = StatUtils.sum(errorFold2.getColumn(0));
                den2 = StatUtils.sum(errorFold2.getColumn(1)); 
                error2 =100*num2/den2;
                //System.out.println("The validation error for the second fold is: "+error2+" %\n\n");
                if(i == 75000-1){
                    System.out.println("The validation error for the second fold is: "+error2+" %\n\n");
                    break;
                }
                //Write to file
                writeToCsv("fold-2.csv",fold2.get(1),output2,min2,max2,error2);
            }
            
            
            //-----------------------------------
            //Train and validate with third fold
            //-----------------------------------
           
            config[2] = fold3.get(0).getRowDimension();
            RealMatrix errorFold3;
            double min3 = StatUtils.min(fold3.get(1).getColumn(4));
            double max3 = StatUtils.max(fold3.get(1).getColumn(4));
            List<RealMatrix> weigthsList3 = createWeigths(vectorJM,nx,M);
            Netcalc.Mw = weigthsList3;
            
            for(int i=0;i<75000;i++){
                errorFold3 = Epoch.computeError(fold3.get(0),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,config);
                double num3 = StatUtils.sum(errorFold3.getColumn(0));
                double den3 = StatUtils.sum(errorFold3.getColumn(1)); 
                //double error3 = (StatUtils.sum(errorFold3.getColumn(2)))/config[2];
                error3 = 100*abs(num3)/abs(den3);
                //System.out.println(i);
                //System.out.println("The error after the training process is: "+ error3+" %");
     
                errorFold3 = Epoch.test(fold3.get(1),Netcalc.Mw, Netcalc.FunctionType.SIGMOID,4,0,133,M,false);
                double[] output3 = errorFold3.getColumn(1);
                for(int j=0;j<output3.length;j++){
                    output3[j] = min3+(max3-min3)*output3[j];
                    //System.out.println(output3[i]);
                }
            num3 = StatUtils.sum(errorFold3.getColumn(0));
            den3 = StatUtils.sum(errorFold3.getColumn(1)); 
            error3 =100*num3/den3;
            //System.out.println("The validation error for the third fold is: "+error3+" %\n\n");
            if(i == 75000-1){
                    System.out.println("The validation error for the third fold is: "+error3+" %\n\n");
                    break;
                }
           
            //Write to file
            writeToCsv("fold-3.csv",fold3.get(1),output3,min3,max3,error3);
            }
            
            
            //Overall error
            System.out.println("The overall error is: "+(error1+error2+error3)/3+"%");
            
        }
        
        //----------------------------------------------------------------------
        //Don't using cross validation
        //----------------------------------------------------------------------
        else{
        
            //----------------------------------------------------------------------
            //Once the weigths have been generated we load the data from the file
            //----------------------------------------------------------------------
            File datafile = new File("Data/A2-ring-merged.txt");
            System.out.println(datafile.getAbsolutePath());
            List<String> listLines = new ArrayList<String>();
            listLines= readLines(datafile,"UTF-8");
            //Store data in a matrix
            RealMatrix dataMatrix = MatrixUtils.createRealMatrix(listLines.size()-1,listLines.get(0).split("\t").length);
            for(int i=0;i<listLines.size()-1;i++){
                for(int j=0;j<listLines.get(0).split("\t").length;j++)
                    dataMatrix.setEntry(i,j,Double.parseDouble(listLines.get(i).split("\t")[j]));            
                }
            //System.out.println(dataMatrix);
            double min = StatUtils.min(dataMatrix.getColumn(2));
            double max = StatUtils.max(dataMatrix.getColumn(2));
            List<RealMatrix> weigthsList = createWeigths(vectorJM,nx,M);
            Netcalc.Mw = weigthsList;
            
            //----------------------------------------------------------------------
            //Training process 
            //----------------------------------------------------------------------
            int[] config = new int[5];
            config[0] = vectorJM[M-2];
            config[1] = nx;
            config[2] = 401;
            config[3] = M;
            RealMatrix errorMatrix;
            double error=0;
        
            for(int i=0;i<150000;i++){
                errorMatrix = Epoch.computeError(dataMatrix,Netcalc.Mw, Netcalc.FunctionType.SIGMOID,config);
                error = (StatUtils.sum(errorMatrix.getColumn(2)))/config[2];
                //System.out.println(error);
            }
            System.out.println("The training error is: "+ error);
            //----------------------------------------------------------------------
            //Test process
            //----------------------------------------------------------------------
            System.out.println("The test process has begin");
            errorMatrix = Epoch.test(dataMatrix,Netcalc.Mw, Netcalc.FunctionType.SIGMOID,4,(451-50),50,M,true);
            double[] output = errorMatrix.getColumn(1);
            for(int i=0;i<output.length;i++){
                output[i] = min+(max-min)*output[i];
                System.out.println(output[i]);
            }
            double num = StatUtils.sum(errorMatrix.getColumn(0));
            double den = StatUtils.sum(errorMatrix.getColumn(1)); 
            error =100*num/den;
            System.out.println("The test error is: "+error+" %");
             
            //Write to file
            writeToCsv("nocrossval.csv",dataMatrix.getSubMatrix(401, 450,4, 4),output,min,max,error);
        }   
    }
    /**
     * Static method to create the three folds to cross validation
     * @param File from where extract the data
     * @param n number of fold
    */
    static List<RealMatrix>  createFold(File dataFile,int n) throws IOException{
        List<String> listLines = new ArrayList<String>();
        List<RealMatrix> dataFold = new ArrayList<RealMatrix>();
        switch(n){
            case 1:
                listLines= readLines(dataFile,"UTF-8");
                //Store data in a matrix
                RealMatrix dataMatrix1 = MatrixUtils.createRealMatrix(listLines.size()-1,listLines.get(0).split(" ").length);
                for(int i=0;i<listLines.size()-1;i++){
                    for(int j=0;j<listLines.get(0).split(" ").length;j++)
                        dataMatrix1.setEntry(i,j,Double.parseDouble(listLines.get(i).split(" ")[j]));            
                    }
                //Train set
                dataFold.add(dataMatrix1.getSubMatrix(0, 267, 0, 4));
                //Validation set
                dataFold.add(dataMatrix1.getSubMatrix(268, 400, 0, 4));
                break;
            case 2:
                listLines= readLines(dataFile,"UTF-8");
                //Store data in a matrix
                RealMatrix dataMatrix2 = MatrixUtils.createRealMatrix(listLines.size()-1,listLines.get(0).split(" ").length);
                for(int i=0;i<listLines.size()-1;i++){
                    for(int j=0;j<listLines.get(0).split(" ").length;j++)
                        dataMatrix2.setEntry(i,j,Double.parseDouble(listLines.get(i).split(" ")[j]));            
                    }
                //Train set 1
                dataFold.add(dataMatrix2.getSubMatrix(0, 133, 0, 4));
                //Validation set
                dataFold.add(dataMatrix2.getSubMatrix(134, 267, 0, 4));
                //Train set 2
                dataFold.add(dataMatrix2.getSubMatrix(268, 400, 0, 4));
                break;
            case 3:
                listLines= readLines(dataFile,"UTF-8");
                //Store data in a matrix
                RealMatrix dataMatrix3 = MatrixUtils.createRealMatrix(listLines.size()-1,listLines.get(0).split(" ").length);
                for(int i=0;i<listLines.size()-1;i++){
                    for(int j=0;j<listLines.get(0).split(" ").length;j++)
                        dataMatrix3.setEntry(i,j,Double.parseDouble(listLines.get(i).split(" ")[j]));            
                    }
                //Train set
                dataFold.add(dataMatrix3.getSubMatrix(134, 400, 0, 4));
                //Validation set 
                dataFold.add(dataMatrix3.getSubMatrix(0, 133, 0, 4));
                break;
            }
        return dataFold;
        }
        /**
        * Static method to create random weigths list
        * @param vectorJM stores the number of neurons of each layer
        * @param nx Number of inputs to the neural network
        * @param M number of layers
        */
        static List<RealMatrix> createWeigths(int vectorJM[],int nx,int M){
            //----------------------------------------------------------------------
            //Define random weigths for each layer to initialize the net
            //----------------------------------------------------------------------
            double leftLimit = -1D;
            double rightLimit = 1D;
            double [][][] matrixWM= new double[M-1][][];
            
            for(int i=0;i<M-1;i++){
            //We generate a random value for each weigth all over the neural net
            var temp = i==0 ? new double[vectorJM[i]][nx+1]:new double[vectorJM[i]][vectorJM[i-1]+1];
            matrixWM [i] = temp;
            Random random = new Random();
            //Weigths for each neuron on each layer
            
                for(int j=0;j<matrixWM[i].length;j++){
                    //Computaion of random weigths for all dendrites each neuron the for lentgh is the number output of the previous layer
                    int wlength = i==0 ? nx+1:vectorJM[i-1]+1;
                    for(int h=0;h<wlength;h++){
                        if(h == 0){
                            //Bias value
                            matrixWM[i][j][0] = 1;
                        }
                        else{
                        //Random uniformly distributed double value between 0 an 1
                        double w = leftLimit + random.nextDouble() * (rightLimit - leftLimit);
                        matrixWM[i][j][h]= w;
                        }
                    }
                }  
            }
            //----------------------------------------------------------------------
            //Extract real matrix for each layer's net
            //----------------------------------------------------------------------
            List<RealMatrix> matrixList = new ArrayList<RealMatrix>();
            for(int i=0;i<M-1;i++){
                RealMatrix m = MatrixUtils.createRealMatrix(matrixWM[i]);
                matrixList.add(m);
            }
        return matrixList;
        }
        static void writeToCsv(String filename,RealMatrix dataMatrix,double[] output,double min,double max,double error) throws IOException{
            //----------------------------------------------------------------------
            //Write output to a file
            //----------------------------------------------------------------------
            File csv = new File("OutputData/"+filename);
            FileWriter fw = new FileWriter("OutputData/"+filename);
            fw.write("original \t predicted\n");
            for(int i=0;i<output.length;i++){
                fw.write(Double.toString(min+(max-min)*(dataMatrix.getEntry(i,dataMatrix.getColumnDimension()-1))));
                fw.write("\t");
                fw.write(Double.toString(output[i]));
                fw.write("\n");
            } 
            fw.write("error\t"+String.valueOf(error)+"%");
            fw.flush();
            fw.close(); 
        }
}
