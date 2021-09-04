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
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

/**
 *Class to compute the ouptput of each neuron
 * @author Miguel Angel Barrero Díaz
 */
    class Netcalc {
        static FunctionSigmoid sigmoid;
        static FunctionSigmoid tanh;
        static List<RealMatrix> Mw;
        static enum FunctionType{
            SIGMOID,
            TANH;
        }
        static{
            System.out.println("Netcalc is computing network parameters...");
        }
        /**
        * Constructor of objects of the class Netcalc
        * @param W  List of RealMatrix elements (all the matrix of weigths of each layer)
        * @param f  Object from the enum class FunctionTypes
        */
        Netcalc(List<RealMatrix> W,FunctionType f){
            Mw = W;
            switch(f){
                case SIGMOID:
                    sigmoid = new FunctionSigmoid();
                break;
                case TANH:
                    tanh = new FunctionSigmoid();
                break;
            }
        }
        /**
        *Constructor to asign high and low asintotes,slope and shift
        * @param W  List of RealMatrix elements (all the matrix of weigths of each layer)
        * @param f  Object from the enum class FunctionTypes
        * @param lo Low asintote of the sigmoid function
        * @param hi High asintote of the sigmoid function
        */
        Netcalc(List<RealMatrix> W,FunctionType f,double lo,double hi,double slope,double threshold){
            Mw = W;
            switch(f){
                case SIGMOID:
                    sigmoid = new FunctionSigmoid(lo,hi,slope,threshold);
                break;
                case TANH:
                    tanh = new FunctionSigmoid();
                break;
            }
        }
        /**
        **Method to do compute the net matrix of each pattern
        * @param pattern Input pattern to analize
        * @param count Set count to zero to control recursion
        * @paran f select function type
        * @return 
        */
        static List<RealVector> computeOutput(RealVector pattern,int count,int layers,FunctionType ftype, List<RealVector> netOutput){
            FunctionSigmoid function = new FunctionSigmoid();
            //We create a list to store matrices to store the net values of each neuron of each layer
            
            int sel = 0;
            switch(ftype){
                case SIGMOID:
                    sel=0;
                    break;
                case TANH:
                    sel=1;
                    break;
           }
            RealMatrix M = Mw.get(count);
            double[] opArray = new double[M.getColumn(0).length];
            RealVector net = MatrixUtils.createRealVector(opArray);
            double bias = 0;
            
           
            //Compute all layers nets using recursion
            for(int i=0;i<M.getColumnVector(0).getDimension();i++){
                RealVector weigths = M.getRowVector(i).getSubVector(1,M.getRowVector(i).getDimension()-1);
                bias = M.getRowVector(i).getEntry(0);
                net.setEntry(i,weigths.dotProduct(pattern));
            }
            
            //Compute sigmoid/tanh
            for(int i=0;i<net.getDimension();i++){
                if(sel==0){
                    opArray[i] = sigmoid.valuesigmoid(net.getEntry(i)-bias);
                    
                }
                else{    
                    opArray[i] = tanh.valuetanh(net.getEntry(i)-bias);
                }
            }
            if(count == 0 ){
                netOutput.add(pattern);
            }
            netOutput.add(count+1, MatrixUtils.createRealVector(opArray));
            //Recursion
            count++;
            
            if(count == layers) return netOutput;
            else netOutput = computeOutput(netOutput.get(netOutput.size()-1),count,layers,FunctionType.SIGMOID,netOutput); return netOutput;
        }
}
