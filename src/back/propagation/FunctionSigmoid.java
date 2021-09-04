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

import org.apache.commons.math3.util.FastMath;

/**
 *
 * @author Miguel Angel Barrero Díaz
 */
public class FunctionSigmoid{
    private final double shift;
    private final double slope;
    private final double lo;
    private final double hi;
    public FunctionSigmoid(double lo,double hi,double slope,double shift){
        this.shift=shift;
        this.slope=slope;
        this.lo = lo;
        this.hi = hi;
    }
    public FunctionSigmoid(){
        super();
        shift = 0;
        slope = 1; 
        this.lo = 0;
        this.hi = 1;
    }
    public double valuesigmoid(double x) {
       
        return (lo + (hi - lo)) / (1 + FastMath.exp(-slope*(x-shift)));
    }
    public double valuetanh(double x){
        return hi *( (FastMath.exp(x)-FastMath.exp(-slope*x+shift))/(FastMath.exp(x)+FastMath.exp(-x+shift)) );
    }
}