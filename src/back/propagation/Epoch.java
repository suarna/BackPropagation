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


import java.util.List;
import back.propagation.Netcalc.FunctionType;
import static back.propagation.Netcalc.FunctionType.SIGMOID;
import java.util.ArrayList;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.stat.StatUtils;
/**
 *
 * @author Miguel Angel Barrero Díaz
 */
public class Epoch {
    /**
    *@param dataMatrix Matrix of data
    *@param matrixList List of matrix of weigths 
    *@param ftype Select the function type to use
    *@param var[0] output length
    *@param var[1] input length
    *@param var[2] trainset size
    *@param var[3] number of layers
    */
    public static RealMatrix computeError(RealMatrix dataMatrix,List<RealMatrix> matrixList,FunctionType ftype,int ... var){
        int tsize = var[2];                         //Train set size
        RealVector e;                               //Difference variable output-desiredoutput
        RealVector output;                          //Output of the net calculator
        RealVector Ep;
        RealMatrix inputMatrix;                     //Matrix of input data patterns
        RealMatrix outputMatrix;                    //Matrix of output data pattern
        RealMatrix eMatrix = MatrixUtils.createRealMatrix(tsize, 1);
        RealMatrix oMatrix = MatrixUtils.createRealMatrix(tsize, 1);
        RealMatrix epMatrix = MatrixUtils.createRealMatrix(tsize, 1);
        RealMatrix error = MatrixUtils.createRealMatrix(tsize, 3);
        int layer = var[3]-1;                       //Layer counter

        //Configure static members of netcalc
        if(ftype==SIGMOID){
            Netcalc.sigmoid = new FunctionSigmoid(0,1,1,0);
        }
        else{
            Netcalc.tanh = new FunctionSigmoid(0,1,1,0);
        }
        //Netcalc.Mw = matrixList;
        
        //----------------------------------------------------------------------
        //FORDWARD PROCESS
        //----------------------------------------------------------------------
       
        //Normalize data
        for(int i=0;i<dataMatrix.getColumnDimension();i++){
            double min = StatUtils.min(dataMatrix.getColumn(i));
            double max = StatUtils.max(dataMatrix.getColumn(i));
            for(int j=0;j<dataMatrix.getRowDimension();j++){
                dataMatrix.setEntry(j, i,(dataMatrix.getEntry(j, i)- min)/(max-min));
            }
        }
       
        //Separte imput data matrix from output data matrix
        inputMatrix = dataMatrix.getSubMatrix(0, dataMatrix.getRowDimension()-1, 0, var[1]-1);
        outputMatrix = dataMatrix.getColumnMatrix(dataMatrix.getColumnDimension()-1);
       
        //System.out.println(outputMatrix);
        //Call to computeNet for each line of the file,i.e.each pattern
        for(int i=0;i<tsize;i++){
            List<RealVector> netOutput = new ArrayList<RealVector>();
            netOutput = Netcalc.computeOutput(inputMatrix.getRowVector(i),0,var[3]-1,ftype,netOutput);
            //Get the output of the last patttern
            output = netOutput.get(netOutput.size()-1);
            //Compute the vector e=(ŷ-y) of the output and the Errors 
            e = output.subtract(outputMatrix.getRowVector(i));
            Ep = (e.ebeMultiply(e).mapMultiply(0.5));  
            eMatrix.setRowVector(i, e); 
            oMatrix.setRowVector(i, output);
            epMatrix.setRowVector(i, Ep);
            
        //----------------------------------------------------------------------
        //BACKWARD PROCESS
        //----------------------------------------------------------------------
        
            double ep;
            double o;
            double wi;
            double biasi;
            double beta = 1;
            double n = 0.003;   
            double alfa = 0.9;
            double[] delta = new double[netOutput.get(layer).getDimension()];
            
            //Output Layer
            if(layer == var[3]-1){
                double incW;
                double incBias;
                double incWprev=0;
                double incBiasprev=0;
                //For all weigths of each neuron in the output of the previous layer
                for(int j=0;j<netOutput.get(layer-1).getDimension();j++){
                    
                    //For each neuron we compute delta System
                    for(int k=0;k<netOutput.get(layer).getDimension();k++){
                        ep = e.getEntry(k);
                        o = netOutput.get(layer).getEntry(k);
                        delta[k] =-ep*beta*o*(1-o);
                        incW = n*delta[k]*(netOutput.get(layer-1).getEntry(j))+delta[k]*incWprev;
                        incBias = n*delta[k]+alfa*incBiasprev;
                        
                        biasi = Netcalc.Mw.get(layer-1).getEntry(k, 0)-incBias;
                        Netcalc.Mw.get(layer-1).setEntry(k, 0, biasi);
                        
                        wi = Netcalc.Mw.get(layer-1).getEntry(k, j+1)+incW;
                        Netcalc.Mw.get(layer-1).setEntry(k, j+1, wi);
                        
                        incWprev = incW;
                        incBiasprev = incBias;
                    }
                }
            }
            layer--;
      
            //Hidden layers
            while(layer> 0){
                double[] delta2 = new double[netOutput.get(layer).getDimension()];
                double incW;
                double incBias;
                double incWprev=0;
                double incBiasprev=0;
                double sum = 0;
                //For all weigths of each neuron in the previous layer
                for(int j=0;j<netOutput.get(layer-1).getDimension();j++){ 
                    //For all neurons of the current layer
                    for(int k=0;k<netOutput.get(layer).getDimension();k++){
                        o = netOutput.get(layer).getEntry(k);
                        //For all the wiegths of ech neuron
                        for(int h=0;h<netOutput.get(layer+1).getDimension();h++){
                            sum = sum +delta[h]*Netcalc.Mw.get(layer).getEntry(h, k);
                        }
                        delta2[k] = sum*beta*o*(1-o);
                        incW = n*delta2[k]*(netOutput.get(layer-1).getEntry(j))+alfa*incWprev;
                        incBias = n*delta2[k]+alfa*incBiasprev;
                        
                        biasi = Netcalc.Mw.get(layer-1).getEntry(k, 0)-incBias;
                        Netcalc.Mw.get(layer-1).setEntry(k,0, biasi);
                        
                        wi = Netcalc.Mw.get(layer-1).getEntry(k, j+1)+incW;
                        Netcalc.Mw.get(layer-1).setEntry(k,j+1, wi);
                        
                        incWprev = incW;
                        incBiasprev = incBias;
                    }
                }
                delta = delta2;
                layer--;
            }
            if(layer == 0){
                layer = var[3]-1;
                }
        }
        error.setColumnMatrix(0, eMatrix);
        error.setColumnMatrix(1, oMatrix);
        error.setColumnMatrix(2, epMatrix);
        return error;
    }
    public static RealMatrix test(RealMatrix dataMatrix,List<RealMatrix> matrixList,FunctionType ftype,int ilength,int firstLine,int tsize,int layers,boolean normalized){
        RealVector e;                               //Difference variable output-desiredoutput
        RealVector output;                          //Output of the net calculator
        RealVector Ep;
        RealMatrix inputMatrix;                     //Matrix of input data patterns
        RealMatrix outputMatrix;                    //Matrix of output data pattern
        RealMatrix eMatrix = MatrixUtils.createRealMatrix(tsize, 1);
        RealMatrix oMatrix = MatrixUtils.createRealMatrix(tsize, 1);
        RealMatrix error = MatrixUtils.createRealMatrix(tsize, 2);
        
        if(normalized == false){
            //Normalize data
            for(int i=0;i<dataMatrix.getColumnDimension();i++){
            double min = StatUtils.min(dataMatrix.getColumn(i));
            double max = StatUtils.max(dataMatrix.getColumn(i));
                for(int j=0;j<dataMatrix.getRowDimension();j++){
                    dataMatrix.setEntry(j, i,(dataMatrix.getEntry(j, i)- min)/(max-min));
                }
            }
        }
        
        //Separte imput data matrix from output data matrix
        inputMatrix = dataMatrix.getSubMatrix(0, dataMatrix.getRowDimension()-1, 0, ilength-1);
        outputMatrix = dataMatrix.getColumnMatrix(dataMatrix.getColumnDimension()-1);
        
        //Call to computeNet for each line of the file,i.e.each pattern
        for(int i=firstLine;i<firstLine+tsize;i++){
            List<RealVector> netOutput = new ArrayList<>();
            netOutput = Netcalc.computeOutput(inputMatrix.getRowVector(i),0,layers-1,ftype,netOutput);
            
            //Get the output of the last patttern
            output = netOutput.get(netOutput.size()-1);
            //Absolute value of the difference
            e = output.subtract(outputMatrix.getRowVector(i));
            for(int j=0;j<e.getDimension();j++){
                if(e.getEntry(j)<0){
                    e.setEntry(j,e.getEntry(j)*(-1));
                }
            }
            Ep = (e.ebeMultiply(e).mapMultiply(0.5));
            eMatrix.setRowVector(i-firstLine, e); 
            oMatrix.setRowVector(i-firstLine, output);
        }
        error.setColumnMatrix(0, eMatrix);
        error.setColumnMatrix(1, oMatrix);
        return error;
    }
}