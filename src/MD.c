/*
 MD.c - a simple molecular dynamics program for simulating real gas properties of Lennard-Jones particles.
 
 Copyright (C) 2016  Jonathan J. Foley IV, Chelsea Sweet, Oyewumi Akinfenwa
 
 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 
 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.
 
 Electronic Contact:  foleyj10@wpunj.edu
 Mail Contact:   Prof. Jonathan Foley
 Department of Chemistry, William Paterson University
 300 Pompton Road
 Wayne NJ 07470
 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdalign.h>
#include <immintrin.h>

//  Lennard-Jones parameters in natural units!
const double sigma = 1.;
const double sigma12 = sigma;
const double sigma6 = sigma;
const double epsilon = 1.;
const double m = 1.;
const double kB = 1.;

const double NA = 6.022140857e23;
const double kBSI = 1.38064852e-23;  // m^2*kg/(s^2*K)


// Maximum particles
const int MAXPART=5001;

typedef struct SimulationValues{
    double pressure, mvs, ke, pe;
}SimulationValues;

typedef __m256d Vec4Double;

Vec4Double v4d_set_all(double v){
    return _mm256_set1_pd(v);
}

Vec4Double v4d_set(double a, double b, double c, double d){
    return _mm256_set_pd(a, b, c, d);
}

double v4d_h_add(Vec4Double a){
    return a[0] + a[1] + a[2] + a[3];
}

Vec4Double v4d_load_u(double *a){
    return _mm256_loadu_pd(a);
}

Vec4Double v4d_packed_add(Vec4Double a, Vec4Double b){
    return _mm256_add_pd(a, b);
}

Vec4Double v4d_packed_sub(Vec4Double a, Vec4Double b){
    return _mm256_sub_pd(a, b);
}

Vec4Double v4d_packed_mul(Vec4Double a, Vec4Double b){
    return _mm256_mul_pd(a, b);
}

Vec4Double v4d_packed_div(Vec4Double a, Vec4Double b){
    return _mm256_div_pd(a, b);
}

void v4d_store_u(Vec4Double a, double b[4]){
    _mm256_storeu_pd(b, a);
}

//  Function prototypes
//  initialize positions on simple cubic lattice, also calls function to initialize velocities
void initialize(int N, double Tinit, double L, double r[][4], double v[][4]);  
//  update positions and velocities using Velocity Verlet algorithm 
//  print particle coordinates to file for rendering via VMD or other animation software
//  return 'instantaneous pressure'
double VelocityVerlet(int N, double L, double dt, FILE *fp, double r[][4], double v[][4], double a[][4]);  
//  Compute Force using F = -dV/dr
//  solve F = ma for use in Velocity Verlet
void computeAccelerations(int N, const double r[][4], double a[][4]);
//  Numerical Recipes function for generation gaussian distribution
double gaussdist();
//  Initialize velocities according to user-supplied initial Temperature (Tinit)
void initializeVelocities(int N, double Tinit, double v[][4]);
//  Compute total potential energy from particle coordinates
double Potential(int N, const double r[][4]);
//  Compute mean squared velocity from particle velocities
double MeanSquaredVelocity(int N, const double v[][4]);
//  Compute total kinetic energy from particle mass and velocities
double Kinetic(int N, const double v[][4]);
SimulationValues simulate(int N, double L, double dt, double r[][4], double v[][4], double a[][4]);

// Calculates power by dividing into multiplication of powers with exponents 1 or multiple of 2.
// Powers are accumulated.
// Ex: 4^7 = 4 * 4^2 * (4^2)^2 = 6 multiplications.
// When inlined, loop is unrolled and branches are removed - tested in gcc 9.5.
double pow_n(double num, unsigned int exp){
    double ret=1, acc=num;
    unsigned int expt=1;
    while(expt<=exp){
        if((expt & exp) == expt)
            ret *= acc;
        acc *= acc;
        expt <<= 1;
    }
    return ret;
}

Vec4Double v4d_pow_n(Vec4Double num, unsigned int exp){
    Vec4Double ret=v4d_set_all(1), acc=num;
    unsigned int expt=1;
    while(expt<=exp){
        if((expt & exp) == expt)
            ret = v4d_packed_mul(ret, acc);
        acc = v4d_packed_mul(acc, acc);
        expt <<= 1;
    }
    return ret;
}

int main()
{
    //  Files and filenames 
    char prefix[1000], tfn[1000], ofn[1000], afn[1000];
    FILE *tfp, *ofp, *afp;
    
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  WELCOME TO WILLY P CHEM MD!\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  ENTER A TITLE FOR YOUR CALCULATION!\n");
    scanf("%s",prefix);
    strcpy(tfn,prefix);
    strcat(tfn,"_traj.xyz");
    strcpy(ofn,prefix);
    strcat(ofn,"_output.txt");
    strcpy(afn,prefix);
    strcat(afn,"_average.txt");
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("                  TITLE ENTERED AS '%s'\n",prefix);
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    
    /*     Table of values for Argon relating natural units to SI units:
     *     These are derived from Lennard-Jones parameters from the article
     *     "Liquid argon: Monte carlo and molecular dynamics calculations"
     *     J.A. Barker , R.A. Fisher & R.O. Watts
     *     Mol. Phys., Vol. 21, 657-673 (1971)
     *
     *     mass:     6.633e-26 kg          = one natural unit of mass for argon, by definition
     *     energy:   1.96183e-21 J      = one natural unit of energy for argon, directly from L-J parameters
     *     length:   3.3605e-10  m         = one natural unit of length for argon, directly from L-J parameters
     *     volume:   3.79499-29 m^3        = one natural unit of volume for argon, by length^3
     *     time:     1.951e-12 s           = one natural unit of time for argon, by length*sqrt(mass/energy)
     ***************************************************************************************/
    
    //  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //  Edit these factors to be computed in terms of basic properties in natural units of
    //  the gas being simulated
    
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("  WHICH NOBLE GAS WOULD YOU LIKE TO SIMULATE? (DEFAULT IS ARGON)\n");
    printf("\n  FOR HELIUM,  TYPE 'He' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR NEON,    TYPE 'Ne' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR ARGON,   TYPE 'Ar' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR KRYPTON, TYPE 'Kr' THEN PRESS 'return' TO CONTINUE\n");
    printf("  FOR XENON,   TYPE 'Xe' THEN PRESS 'return' TO CONTINUE\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    // atom type
    char atype[10];

    scanf("%s",atype);
    
    double VolFac, TempFac, PressFac, timefac;
    if (strcmp(atype,"He")==0) {
        
        VolFac = 1.8399744000000005e-29;
        PressFac = 8152287.336171632;
        TempFac = 10.864459551225972;
        timefac = 1.7572698825166272e-12;
        
    }
    else if (strcmp(atype,"Ne")==0) {
        
        VolFac = 2.0570823999999997e-29;
        PressFac = 27223022.27659913;
        TempFac = 40.560648991243625;
        timefac = 2.1192341945685407e-12;
        
    }
    else if (strcmp(atype,"Ar")==0) {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        //strcpy(atype,"Ar");
        
    }
    else if (strcmp(atype,"Kr")==0) {
        
        VolFac = 4.5882712000000004e-29;
        PressFac = 59935428.40275003;
        TempFac = 199.1817584391428;
        timefac = 8.051563913585078e-13;
        
    }
    else if (strcmp(atype,"Xe")==0) {
        
        VolFac = 5.4872e-29;
        PressFac = 70527773.72794868;
        TempFac = 280.30305642163006;
        timefac = 9.018957925790732e-13;
        
    }
    else {
        
        VolFac = 3.7949992920124995e-29;
        PressFac = 51695201.06691862;
        TempFac = 142.0950000000000;
        timefac = 2.09618e-12;
        strcpy(atype,"Ar");
        
    }
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n                     YOU ARE SIMULATING %s GAS! \n",atype);
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    
    printf("\n  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n  YOU WILL NOW ENTER A FEW SIMULATION PARAMETERS\n");
    printf("  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    printf("\n\n  ENTER THE INTIAL TEMPERATURE OF YOUR GAS IN KELVIN\n");

    //  Initial Temperature in Natural Units
    double Tinit;  //2;

    scanf("%lf",&Tinit);
    // Make sure temperature is a positive number!
    if (Tinit<0.) {
        printf("\n  !!!!! ABSOLUTE TEMPERATURE MUST BE A POSITIVE NUMBER!  PLEASE TRY AGAIN WITH A POSITIVE TEMPERATURE!!!\n");
        exit(0);
    }
    // Convert initial temperature from kelvin to natural units
    Tinit /= TempFac;
    
    
    printf("\n\n  ENTER THE NUMBER DENSITY IN moles/m^3\n");
    printf("  FOR REFERENCE, NUMBER DENSITY OF AN IDEAL GAS AT STP IS ABOUT 40 moles/m^3\n");
    printf("  NUMBER DENSITY OF LIQUID ARGON AT 1 ATM AND 87 K IS ABOUT 35000 moles/m^3\n");
    
    double rho;
    scanf("%lf",&rho);
    
    // Number of particles
    int N = 10*216;

    double Vol = N/(rho*NA);
    
    Vol /= VolFac;
    
    //  Limiting N to MAXPART for practical reasons
    if (N>=MAXPART) {
        
        printf("\n\n\n  MAXIMUM NUMBER OF PARTICLES IS %i\n\n  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY \n\n", MAXPART);
        exit(0);
        
    }
    //  Check to see if the volume makes sense - is it too small?
    //  Remember VDW radius of the particles is 1 natural unit of length
    //  and volume = L*L*L, so if V = N*L*L*L = N, then all the particles
    //  will be initialized with an interparticle separation equal to 2xVDW radius
    if (Vol<N) {
        
        printf("\n\n\n  YOUR DENSITY IS VERY HIGH!\n\n");
        printf("  THE NUMBER OF PARTICLES IS %i AND THE AVAILABLE VOLUME IS %f NATURAL UNITS\n",N,Vol);
        printf("  SIMULATIONS WITH DENSITY GREATER THAN 1 PARTCICLE/(1 Natural Unit of Volume) MAY DIVERGE\n");
        printf("  PLEASE ADJUST YOUR INPUT FILE ACCORDINGLY AND RETRY\n\n");
        exit(0);
    }
    // Vol = L*L*L;
    // Length of the box in natural units:
    double L = pow(Vol,(1./3));
    
    //  Files that we can write different quantities to
    tfp = fopen(tfn,"w");     //  The MD trajectory, coordinates of every particle at each timestep
    ofp = fopen(ofn,"w");     //  Output of other quantities (T, P, gc, etc) at every timestep
    afp = fopen(afn,"w");    //  Average T, P, gc, etc from the simulation
    
    int NumTime;
    double dt;
    if (strcmp(atype,"He")==0) {
        
        // dt in natural units of time s.t. in SI it is 5 f.s. for all other gasses
        dt = 0.2e-14/timefac;
        //  We will run the simulation for NumTime timesteps.
        //  The total time will be NumTime*dt in natural units
        //  And NumTime*dt multiplied by the appropriate conversion factor for time in seconds
        NumTime=50000;
    }
    else {
        dt = 0.5e-14/timefac;
        NumTime=200;
        
    }
    
    //  Position
    double r[MAXPART][4];
    //  Velocity
    double v[MAXPART][4];
    //  Acceleration
    double a[MAXPART][4];

    //  Put all the atoms in simple crystal lattice and give them random velocities
    //  that corresponds to the initial temperature we have specified
    initialize(N, Tinit, L, r, v);
    
    //  Based on their positions, calculate the ininial intermolecular forces
    //  The accellerations of each particle will be defined from the forces and their
    //  mass, and this will allow us to update their positions via Newton's law
    computeAccelerations(N, r, a);
    
    
    // Print number of particles to the trajectory file
    fprintf(tfp,"%i\n",N);
    
    //  We want to calculate the average Temperature and Pressure for the simulation
    //  The variables need to be set to zero initially
    double Pavg = 0;
    double Tavg = 0;
    
    
    fprintf(ofp,"  time (s)              T(t) (K)              P(t) (Pa)           Kinetic En. (n.u.)     Potential En. (n.u.) Total En. (n.u.)\n");
    printf("  SIMULATING...:\n\n");

    struct SimulationResult {
        double time;
        double temperature;
        double pressure;
        double kineticEnergy;
        double potentialEnergy;
        double totalEnergy;
    };

    double gc, Z;
    struct SimulationResult results[NumTime]; // Max is 50000


    for (int i=0; i <= NumTime; i++) {

        

        // This updates the positions and velocities using Newton's Laws.
        // Computes the Pressure as the sum of momentum changes from wall collisions / timestep
        // which is a Kinetic Theory of gasses concept of Pressure.
        // Also computes Instantaneous mean velocity squared, Potential and Kinetic Energy.
        SimulationValues vals = simulate(N, L, dt, r, v, a);
        vals.pressure *= PressFac;

        // Temperature from Kinetic Theory
        double Temp = m*vals.mvs/(3*kB) * TempFac;
        
        // Instantaneous gas constant and compressibility - not well defined because
        // pressure may be zero in some instances because there will be zero wall collisions,
        // pressure may be very high in some instances because there will be a number of collisions
        gc = NA*vals.pressure*(Vol*VolFac)/(N*Temp);
        Z  = vals.pressure*(Vol*VolFac)/(N*kBSI*Temp);
        
        Tavg += Temp;
        Pavg += vals.pressure;
        
        // Store the results in the struct
        results[i].time = i * dt * timefac;
        results[i].temperature = Temp;
        results[i].pressure = vals.pressure;
        results[i].kineticEnergy = vals.ke;
        results[i].potentialEnergy = vals.pe;
        results[i].totalEnergy = vals.ke + vals.pe;
    }

    for (int j = 0; j <= NumTime; j++) {
        fprintf(ofp, "  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n",
        results[j].time, results[j].temperature, results[j].pressure,
        results[j].kineticEnergy, results[j].potentialEnergy, results[j].totalEnergy);
    }

    // Because we have calculated the instantaneous temperature and pressure,
    // we can take the average over the whole simulation here
    Pavg /= NumTime;
    Tavg /= NumTime;
    Z = Pavg*(Vol*VolFac)/(N*kBSI*Tavg);
    gc = NA*Pavg*(Vol*VolFac)/(N*Tavg);
    fprintf(afp,"  Total Time (s)      T (K)               P (Pa)      PV/nT (J/(mol K))         Z           V (m^3)              N\n");
    fprintf(afp," --------------   -----------        ---------------   --------------   ---------------   ------------   -----------\n");
    fprintf(afp,"  %8.4e  %15.5f       %15.5f     %10.5f       %10.5f        %10.5e         %i\n",(NumTime+1)*dt*timefac,Tavg,Pavg,gc,Z,Vol*VolFac,N);
    
    printf("\n  TO ANIMATE YOUR SIMULATION, OPEN THE FILE \n  '%s' WITH VMD AFTER THE SIMULATION COMPLETES\n",tfn);
    printf("\n  TO ANALYZE INSTANTANEOUS DATA ABOUT YOUR MOLECULE, OPEN THE FILE \n  '%s' WITH YOUR FAVORITE TEXT EDITOR OR IMPORT THE DATA INTO EXCEL\n",ofn);
    printf("\n  THE FOLLOWING THERMODYNAMIC AVERAGES WILL BE COMPUTED AND WRITTEN TO THE FILE  \n  '%s':\n",afn);
    printf("\n  AVERAGE TEMPERATURE (K):                 %15.5f\n",Tavg);
    printf("\n  AVERAGE PRESSURE  (Pa):                  %15.5f\n",Pavg);
    printf("\n  PV/nT (J * mol^-1 K^-1):                 %15.5f\n",gc);
    printf("\n  PERCENT ERROR of pV/nT AND GAS CONSTANT: %15.5f\n",100*fabs(gc-8.3144598)/8.3144598);
    printf("\n  THE COMPRESSIBILITY (unitless):          %15.5f \n",Z);
    printf("\n  TOTAL VOLUME (m^3):                      %10.5e \n",Vol*VolFac);
    printf("\n  NUMBER OF PARTICLES (unitless):          %i \n", N);
    
    
    
    
    fclose(tfp);
    fclose(ofp);
    fclose(afp);
    
    return 0;
}


void initialize(int N, double Tinit, double L, double r[][4], double v[][4]) {
    int n, p, i, j, k;
    double pos;
    
    // Number of atoms in each direction
    n = (int)ceil(pow(N, 1.0/3));
    
    //  spacing between atoms along a given direction
    pos = L / n;
    
    //  index for number of particles assigned positions
    p = 0;
    //  initialize positions
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            for (k=0; k<n; k++) {
                if (p<N) {
                    
                    r[p][0] = (i + 0.5)*pos;
                    r[p][1] = (j + 0.5)*pos;
                    r[p][2] = (k + 0.5)*pos;
                }
                p++;
            }
        }
    }
    
    // Call function to initialize velocities
    initializeVelocities(N, Tinit, v);
    
    /***********************************************
     *   Uncomment if you want to see what the initial positions and velocities are
     printf("  Printing initial positions!\n");
     for (i=0; i<N; i++) {
     printf("  %6.3e  %6.3e  %6.3e\n",r[i][0],r[i][1],r[i][2]);
     }
     
     printf("  Printing initial velocities!\n");
     for (i=0; i<N; i++) {
     printf("  %6.3e  %6.3e  %6.3e\n",v[i][0],v[i][1],v[i][2]);
     }
     */
    
    
    
}   


//  Function to calculate the averaged velocity squared
double MeanSquaredVelocity(int N, const double v[][4]) { 
    
    double vx2 = 0;
    double vy2 = 0;
    double vz2 = 0;
    double v2;
    
    for (int i=0; i<N; i++) {
        
        vx2 = vx2 + v[i][0]*v[i][0];
        vy2 = vy2 + v[i][1]*v[i][1];
        vz2 = vz2 + v[i][2]*v[i][2];
        
    }
    v2 = (vx2+vy2+vz2)/N;
    
    
    //printf("  Average of x-component of velocity squared is %f\n",v2);
    return v2;
}

//  Function to calculate the kinetic energy of the system
double Kinetic(int N, const double v[][4]) { 
    
    double v2, kin;
    
    kin =0.;
    for (int i=0; i<N; i++) {
        
        v2 = 0.;
        for (int j=0; j<3; j++) {
            
            v2 += v[i][j]*v[i][j];
            
        }
        kin += m*v2/2.;
        
    }
    
    //printf("  Total Kinetic Energy is %f\n",N*mvs*m/2.);
    return kin;
    
}


// Function to calculate the potential energy of the system
double Potential(int N, const double r[][4]) {
    double Pot=0.;

    for (int i=0; i<N-1; i++) {
        for (int j=i+1; j<N; j++) {
            double r2=0.;
            for (int k=0; k<3; k++) {
                double delta_r = r[i][k] - r[j][k];
                r2 += delta_r * delta_r;
            }
            double r2p3 = pow_n(r2,3);
            
            Pot += 2*4*epsilon*(sigma12/r2p3 - sigma6)/r2p3;
        }
    }
    
    return Pot;
}

//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
void computeAccelerations(int N, const double r[][4], double a[][4]) {

    // set all accelerations to zero
    memset(a, 0, N*3*sizeof(double));

    for (int i = 0; i < N-1; i++) {   // loop over all distinct pairs i,j
        for (int j = i+1; j < N; j++) {
            // initialize r^2 to zero
            double rSqd = 0;
            double rij[3]; // position of i relative to j
            
            for (int k = 0; k < 3; k++) {
                //  component-by-componenent position of i relative to j
                rij[k] = r[i][k] - r[j][k];
                //  sum of squares of the components
                rSqd += rij[k] * rij[k];
            }
            
            //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            double f = (48 - 24*pow_n(rSqd, 3)) / pow_n(rSqd, 7);
    
            for (int k = 0; k < 3; k++) {
                //  from F = ma, where m = 1 in natural units!
                a[i][k] += rij[k] * f;
                a[j][k] -= rij[k] * f;
            }
        }
    }
}


// returns sum of dv/dt*m/A (aka Pressure) from elastic collisions with walls
double VelocityVerlet(int N, double L, double dt, FILE *fp, double r[][4], double v[][4], double a[][4]) {
    double psum = 0.;
    
    //  Update positions and velocity with current velocity and acceleration
    for (int i=0; i<N; i++) {
        for (int j=0; j<3; j++) {
            v[i][j] += 0.5*a[i][j]*dt;
            r[i][j] += v[i][j]*dt;
        }
    }

    //  Update accellerations from updated positions
    computeAccelerations(N, r, a);
    
    // Elastic walls
    for (int i=0; i<N; i++) {
        for (int j=0; j<3; j++) {
            v[i][j] += 0.5*a[i][j]*dt; //  Update velocity with updated acceleration

            if (r[i][j]<0. || r[i][j]>=L) {
                v[i][j] *=-1.; //- elastic walls
                psum += 2*m*fabs(v[i][j])/dt;  // contribution to pressure from "left" walls
            }
        }
    }

    return psum/(6*L*L);
}


double computeAccelerationsAndPotential(int N, double r[][4], double a[][4]){
    // set all accelerations to zero
    memset(a, 0, N*4*sizeof(double));
    Vec4Double potential = v4d_set_all(0.0);
    double pot_last_iter = 0.0;

    for (int i = 0; i < N-1; i++) {   // loop over all distinct pairs i,j
        for (int j = i+1; j < N-((N-(i+1))%4); j+=4) {
            Vec4Double pos_i = v4d_load_u(r[i]);
            Vec4Double pos_j0 = v4d_load_u(r[j]);
            Vec4Double pos_j1 = v4d_load_u(r[j+1]);
            Vec4Double pos_j2 = v4d_load_u(r[j+2]);
            Vec4Double pos_j3 = v4d_load_u(r[j+3]);
            
            //  distance of i relative to j
            Vec4Double rij0 = pos_i - pos_j0;
            Vec4Double rij1 = pos_i - pos_j1;
            Vec4Double rij2 = pos_i - pos_j2;
            Vec4Double rij3 = pos_i - pos_j3;

            // dot product of distance
            Vec4Double dp;
            {
                Vec4Double mult1 = rij0 * rij0;
                Vec4Double mult2 = rij1 * rij1;
                Vec4Double mult3 = rij2 * rij2;
                Vec4Double mult4 = rij3 * rij3;

                Vec4Double add1 = {mult1[0], mult2[0], mult3[0], mult4[0]};
                Vec4Double add2 = {mult1[1], mult2[1], mult3[1], mult4[1]};
                Vec4Double add3 = {mult1[2], mult2[2], mult3[2], mult4[2]};

                dp = (add1 + add2) + add3;
            }

            // Compute Potential
            Vec4Double dp3 = v4d_pow_n(dp, 3);
            {
                Vec4Double eps_const = v4d_set_all(8*epsilon);
                Vec4Double sigma12_v = v4d_set_all(sigma12), sigma6_v = v4d_set_all(sigma6);

                potential += eps_const*(sigma12_v/dp3 - sigma6_v)/dp3;
            }
            
            //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            Vec4Double dp7 = v4d_pow_n(dp, 7);
            Vec4Double vec_48 = v4d_set_all(48.0), vec_24 = v4d_set_all(24.0);

            Vec4Double forces = v4d_packed_mul(vec_24, dp3);
            forces = v4d_packed_sub(vec_48, forces);
            forces = v4d_packed_div(forces, dp7);

            Vec4Double f0=v4d_set_all(forces[0]),f1=v4d_set_all(forces[1]),
                        f2=v4d_set_all(forces[2]),f3=v4d_set_all(forces[3]);

            rij0 = v4d_packed_mul(rij0, f0);
            rij1 = v4d_packed_mul(rij1, f1);
            rij2 = v4d_packed_mul(rij2, f2);
            rij3 = v4d_packed_mul(rij3, f3);

            //  from F = ma, where m = 1 in natural units!
            Vec4Double accel_i = v4d_load_u(a[i]);
            Vec4Double accel_j0 = v4d_load_u(a[j]);
            Vec4Double accel_j1 = v4d_load_u(a[j+1]);
            Vec4Double accel_j2 = v4d_load_u(a[j+2]);
            Vec4Double accel_j3 = v4d_load_u(a[j+3]);

            accel_i = v4d_packed_add(accel_i, rij0);
            accel_i = v4d_packed_add(accel_i, rij1);
            accel_i = v4d_packed_add(accel_i, rij2);
            accel_i = v4d_packed_add(accel_i, rij3);

            accel_j0 = v4d_packed_sub(accel_j0, rij0);
            accel_j1 = v4d_packed_sub(accel_j1, rij1);
            accel_j2 = v4d_packed_sub(accel_j2, rij2);
            accel_j3 = v4d_packed_sub(accel_j3, rij3);

            v4d_store_u(accel_i, a[i]);
            v4d_store_u(accel_j0, a[j]);
            v4d_store_u(accel_j1, a[j+1]);
            v4d_store_u(accel_j2, a[j+2]);
            v4d_store_u(accel_j3, a[j+3]);
        }
        for (int j = N-(N-(i+1))%4; j < N; j++) {
            double rij[3]; // distance of i relative to j
            
            //  distance of i relative to j
            rij[0] = r[i][0] - r[j][0];
            rij[1] = r[i][1] - r[j][1];
            rij[2] = r[i][2] - r[j][2];

            //  dot product of distance
            double rSqd = (rij[0] * rij[0])+(rij[1] * rij[1])+(rij[2] * rij[2]);

            // Compute Potential
            double r2p3 = pow_n(rSqd,3);
            pot_last_iter += 2*4*epsilon*(sigma12/r2p3 - sigma6)/r2p3;
            
            //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            double f = (48 - 24*r2p3) / pow_n(rSqd, 7);

            rij[0] *= f; 
            rij[1] *= f; 
            rij[2] *= f; 

            //  from F = ma, where m = 1 in natural units!
            a[i][0] += rij[0];
            a[i][1] += rij[1];
            a[i][2] += rij[2];

            a[j][0] -= rij[0];
            a[j][1] -= rij[1];
            a[j][2] -= rij[2];
        }
    }

    return potential[0]+potential[1]+potential[2]+potential[3]+pot_last_iter;
}

//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
double computeAccelerationsAndPotential2(int N, const double r[][4], double a[][4]){
    // set all accelerations to zero
    memset(a, 0, N*4*sizeof(double));
    double potential=0.;

    for (int i = 0; i < N-1; i++) {   // loop over all distinct pairs i,j
        for (int j = i+1; j < N; j++) {
            double rij[3]; // distance of i relative to j
            
            //  distance of i relative to j
            rij[0] = r[i][0] - r[j][0];
            rij[1] = r[i][1] - r[j][1];
            rij[2] = r[i][2] - r[j][2];

            //  dot product of distance
            double rSqd = (rij[0] * rij[0])+(rij[1] * rij[1])+(rij[2] * rij[2]);

            // Compute Potential
            double r2p3 = pow_n(rSqd,3);
            potential += 2*4*epsilon*(sigma12/r2p3 - sigma6)/r2p3;
            
            //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            double f = (48 - 24*r2p3) / pow_n(rSqd, 7);

            rij[0] *= f; 
            rij[1] *= f; 
            rij[2] *= f; 

            //  from F = ma, where m = 1 in natural units!
            a[i][0] += rij[0];
            a[i][1] += rij[1];
            a[i][2] += rij[2];

            a[j][0] -= rij[0];
            a[j][1] -= rij[1];
            a[j][2] -= rij[2];
        }
            printf("%f\n", potential);
            if(i==1) exit(0);
    }
    return potential;
}

SimulationValues simulate(int N, double L, double dt, double r[][4], double v[][4], double a[][4]) {
    double psum = 0.;
    
    //  Update positions and velocity with current velocity and acceleration
    for (int i=0; i<N; i++) {
        for (int j=0; j<3; j++) {
            v[i][j] += 0.5*a[i][j]*dt;
            r[i][j] += v[i][j]*dt;
        }
    }
    //  Update accellerations from updated positions and compute potential energy
    double pe = computeAccelerationsAndPotential(N, r, a);
    
    // Elastic walls
    for (int i=0; i<N; i++) {
        for (int j=0; j<3; j++) {
            v[i][j] += 0.5*a[i][j]*dt; //  Update velocity with updated acceleration

            if (r[i][j]<0. || r[i][j]>=L) {
                v[i][j] *=-1.; //- elastic walls
                psum += 2*m*fabs(v[i][j])/dt;  // contribution to pressure from "left" walls
            }
        }
    }

    double mvs = MeanSquaredVelocity(N, v);
    double ke = Kinetic(N, v);

    SimulationValues values = {
        .pressure = psum/(6*L*L),
        .pe = pe,
        .ke = ke,
        .mvs = mvs,
    };

    return values;
}


void initializeVelocities(int N, double Tinit, double v[][4]) {
    
    int i, j;
    
    for (i=0; i<N; i++) {
        
        for (j=0; j<3; j++) {
            //  Pull a number from a Gaussian Distribution
            v[i][j] = gaussdist();
            
        }
    }
    
    // Vcm = sum_i^N  m*v_i/  sum_i^N  M
    // Compute center-of-mas velocity according to the formula above
    double vCM[3] = {0, 0, 0};
    
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            
            vCM[j] += m*v[i][j];
            
        }
    }
    
    
    for (i=0; i<3; i++) vCM[i] /= N*m;
    
    //  Subtract out the center-of-mass velocity from the
    //  velocity of each particle... effectively set the
    //  center of mass velocity to zero so that the system does
    //  not drift in space!
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            
            v[i][j] -= vCM[j];
            
        }
    }
    
    //  Now we want to scale the average velocity of the system
    //  by a factor which is consistent with our initial temperature, Tinit
    double vSqdSum, lambda;
    vSqdSum=0.;
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            
            vSqdSum += v[i][j]*v[i][j];
            
        }
    }
    
    lambda = sqrt( 3*(N-1)*Tinit/vSqdSum);
    
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            
            v[i][j] *= lambda;
            
        }
    }
}


//  Numerical recipes Gaussian distribution number generator
double gaussdist() {
    static bool available = false;
    static double gset;
    double fac, rsq, v1, v2;
    if (!available) {
        do {
            v1 = 2.0 * rand() / (double)(RAND_MAX) - 1.0;
            v2 = 2.0 * rand() / (double)(RAND_MAX) - 1.0;
            rsq = v1 * v1 + v2 * v2;
        } while (rsq >= 1.0 || rsq == 0.0);
        
        fac = sqrt(-2.0 * log(rsq) / rsq);
        gset = v1 * fac;
        available = true;
        
        return v2*fac;
    } else {
        
        available = false;
        return gset;
        
    }
}
