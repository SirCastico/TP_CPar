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

// vector types using vector extension, supported by gcc and clang
typedef double __attribute__((vector_size(32), aligned(32))) v4df;
typedef double __attribute__((vector_size(32), aligned(1))) __v4df_u; // internal unaligned type

v4df v4df_set(double a, double b, double c, double d){
    return (v4df){d,c,b,a};
}

v4df v4df_set_all(double v){
    return v4df_set(v,v,v,v);
}

double v4df_h_add(v4df a){
    return a[0] + a[1] + a[2] + a[3];
}

v4df v4df_load(double *a){
    return *(v4df*)a;
}

void v4df_store(v4df a, double *p){
  *(v4df *)p = a;
}

v4df v4df_load_u(double *a){
    // from clang implementation of unaligned load (compatible with gcc)
    struct __loadu_pd {
    __v4df_u __v;
    } __attribute__((__packed__, __may_alias__));
    return ((const struct __loadu_pd*)a)->__v;
}

void v4df_store_u(v4df a, double b[4]){
    // form clang implementation of unaligned store (compatible with gcc)
    struct __storeu_pd {
    __v4df_u __v;
    } __attribute__((__packed__, __may_alias__));
    ((struct __storeu_pd*)b)->__v = a;
}

//  Function prototypes
//  initialize positions on simple cubic lattice, also calls function to initialize velocities
void initialize(int N, double Tinit, double L, double r[3][MAXPART], double v[3][MAXPART]);  
//  update positions and velocities using Velocity Verlet algorithm 
//  print particle coordinates to file for rendering via VMD or other animation software
//  return 'instantaneous pressure'
double VelocityVerlet(int N, double L, double dt, FILE *fp, double r[3][MAXPART], double v[3][MAXPART], double a[3][MAXPART]);  
//  Compute Force using F = -dV/dr
//  solve F = ma for use in Velocity Verlet
void computeAccelerations(int N, const double r[3][MAXPART], double a[3][MAXPART]);
//  Numerical Recipes function for generation gaussian distribution
double gaussdist();
//  Initialize velocities according to user-supplied initial Temperature (Tinit)
void initializeVelocities(int N, double Tinit, double v[3][MAXPART]);
//  Compute total potential energy from particle coordinates
double Potential(int N, const double r[3][MAXPART]);
//  Compute mean squared velocity from particle velocities
double MeanSquaredVelocity(int N, const double v[3][MAXPART]);
//  Compute total kinetic energy from particle mass and velocities
double Kinetic(int N, const double v[3][MAXPART]);
SimulationValues simulate(int N, double L, double dt, double r[3][MAXPART], double v[3][MAXPART], double a[3][MAXPART]);

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

v4df v4df_pow_n(v4df num, unsigned int exp){
    v4df ret=v4df_set_all(1), acc=num;
    unsigned int expt=1;
    while(expt<=exp){
        if((expt & exp) == expt)
            ret = ret * acc;
        acc = acc * acc;
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
    alignas(32) double r[3][MAXPART];
    //  Velocity
    alignas(32) double v[3][MAXPART];
    //  Acceleration
    alignas(32) double a[3][MAXPART];

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


void initialize(int N, double Tinit, double L, double r[3][MAXPART], double v[3][MAXPART]) {
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
                    
                    r[0][p] = (i + 0.5)*pos;
                    r[1][p] = (j + 0.5)*pos;
                    r[2][p] = (k + 0.5)*pos;
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
double MeanSquaredVelocity(int N, const double v[3][MAXPART]) { 
    
    double vx2 = 0;
    double vy2 = 0;
    double vz2 = 0;
    double v2;
    
    for (int i=0; i<N; i++) {
        
        vx2 = vx2 + v[0][i]*v[0][i];
        vy2 = vy2 + v[1][i]*v[1][i];
        vz2 = vz2 + v[2][i]*v[2][i];
        
    }
    v2 = (vx2+vy2+vz2)/N;
    
    
    //printf("  Average of x-component of velocity squared is %f\n",v2);
    return v2;
}

//  Function to calculate the kinetic energy of the system
double Kinetic(int N, const double v[3][MAXPART]) { 
    
    double v2, kin;
    
    kin =0.;
    for (int j=0; j<3; j++) {
        
        v2 = 0.;
        for (int i=0; i<N; i++) {
            
            v2 += v[j][i]*v[j][i];
            
        }
        kin += m*v2/2.;
        
    }
    
    //printf("  Total Kinetic Energy is %f\n",N*mvs*m/2.);
    return kin;
    
}


//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
void computeAccelerations(int N, const double r[3][MAXPART], double a[3][MAXPART]) {

    // set all accelerations to zero
    memset(a, 0, MAXPART*3*sizeof(double));

    for (int i = 0; i < N-1; i++) {   // loop over all distinct pairs i,j
        for (int j = i+1; j < N; j++) {
            // initialize r^2 to zero
            double rij[3]; // position of i relative to j

            //  distance of i relative to j
            rij[0] = r[0][i] - r[0][j];
            rij[1] = r[1][i] - r[1][j];
            rij[2] = r[2][i] - r[2][j];
            
            double rSqd = (rij[0] * rij[0])+(rij[1] * rij[1])+(rij[2] * rij[2]);

            //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
            double f = (48 - 24*pow_n(rSqd, 3)) / pow_n(rSqd, 7);

            rij[0] *= f; 
            rij[1] *= f; 
            rij[2] *= f; 

            //  from F = ma, where m = 1 in natural units!
            a[0][i] += rij[0];
            a[1][i] += rij[1];
            a[2][i] += rij[2];

            a[0][j] -= rij[0];
            a[1][j] -= rij[1];
            a[2][j] -= rij[2];
    
        }
    }
}


//   Uses the derivative of the Lennard-Jones potential to calculate
//   the forces on each atom.  Then uses a = F/m to calculate the
//   accelleration of each atom. 
//   Also calculates and returns potential energy of the system.
double computeAccelerationsAndPotential(int N, double r[3][MAXPART], double a[3][MAXPART]){
    // set all accelerations to zero
    memset(a, 0, MAXPART*3*sizeof(double));

    // vector to accumulate the potential calculations
    v4df potential = v4df_set_all(0.0);
    double pot_last_iter = 0.0;

    // setup constants
    v4df eps_const = v4df_set_all(8*epsilon);
    v4df sigma12_v = v4df_set_all(sigma12), sigma6_v = v4df_set_all(sigma6);
    v4df vec_48 = v4df_set_all(48.0), vec_24 = v4df_set_all(24.0);

    for (int i = 0; i < N-1; i++) {   // loop over all distinct pairs i,j, 4 j particles at a time
        double pos_i[3] = {r[0][i], r[1][i], r[2][i]};

        // repeat each particle i coordinate into a different vector 
        v4df vpos_ix = v4df_set_all(pos_i[0]), vpos_iy = v4df_set_all(pos_i[1]), vpos_iz = v4df_set_all(pos_i[2]);

        // setup particle i acceleration accumulators, storing every acceleration computation affecting particle i.
        // coordinates of same dimension are stored on the same vector.
        v4df vaccel_ix_acc = v4df_set_all(0), vaccel_iy_acc = v4df_set_all(0), vaccel_iz_acc = v4df_set_all(0);

        for (int j = i+1; j < N-((N-(i+1))%4); j+=4) {
            // load j,j+1,j+2,j+3 positions, coordinates of same dimension stored on the same vector
            v4df pos_jx = v4df_load_u(&r[0][j]);
            v4df pos_jy = v4df_load_u(&r[1][j]);
            v4df pos_jz = v4df_load_u(&r[2][j]);
            
            //  distance of i relative to j,j+1,j+2,j+3, coordinates of same dimension stored on the same vector
            v4df dist_x = vpos_ix - pos_jx;
            v4df dist_y = vpos_iy - pos_jy;
            v4df dist_z = vpos_iz - pos_jz;

            // dot product of distance of the 4 pairs
            // dot product for different pairs stored in different vector slots
            v4df dp = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

            // Compute Potential of the 4 pairs
            v4df dp3 = v4df_pow_n(dp, 3);
            {
                v4df dp3_div = 1/dp3;

                potential += eps_const*(sigma12_v*dp3_div - sigma6_v)*dp3_div;
            }
            
            // Compute Accelerations of the 4 pairs
            {
                v4df dp7 = v4df_pow_n(dp, 7);

                //  From derivative of Lennard-Jones with sigma and epsilon set equal to 1 in natural units!
                v4df forces = (vec_48 - vec_24 * dp3)/dp7;

                //  from F = ma, where m = 1 in natural units!
                dist_x *= forces;
                dist_y *= forces;
                dist_z *= forces;

                // load j,j+1,j+2,j+3 accelerations, coordinates of same dimension stored on the same vector
                v4df accel_jx = v4df_load_u(&a[0][j]);
                v4df accel_jy = v4df_load_u(&a[1][j]);
                v4df accel_jz = v4df_load_u(&a[2][j]);

                // accumulate particle i acceleration
                vaccel_ix_acc += dist_x;
                vaccel_iy_acc += dist_y;
                vaccel_iz_acc += dist_z;

                // accumulate particle j,j+1,j+2,j+3 acceleration
                accel_jx -= dist_x;
                accel_jy -= dist_y;
                accel_jz -= dist_z;

                v4df_store_u(accel_jx, &a[0][j]);
                v4df_store_u(accel_jy, &a[1][j]);
                v4df_store_u(accel_jz, &a[2][j]);
            }
        }
        double accel_i_acc[3] = {0,0,0};
        for (int j = N-(N-(i+1))%4; j < N; j++) {
            double rij[3]; // distance of i relative to j
            
            //  distance of i relative to j
            rij[0] = pos_i[0] - r[0][j];
            rij[1] = pos_i[1] - r[1][j];
            rij[2] = pos_i[2] - r[2][j];

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
            accel_i_acc[0] += rij[0];
            accel_i_acc[1] += rij[1];
            accel_i_acc[2] += rij[2];

            a[0][j] -= rij[0];
            a[1][j] -= rij[1];
            a[2][j] -= rij[2];
        }
        // store particle i accelerations
        a[0][i] += accel_i_acc[0] + v4df_h_add(vaccel_ix_acc);
        a[1][i] += accel_i_acc[1] + v4df_h_add(vaccel_iy_acc);
        a[2][i] += accel_i_acc[2] + v4df_h_add(vaccel_iz_acc);
    }

    return v4df_h_add(potential)+pot_last_iter;
}


// Executes a step in the simulation and updates the particles' properties.
// Returns pressure from collisions with elastic walls, potential energy, 
// mean squared velocity of particles and the total kinetic energy. 
SimulationValues simulate(int N, double L, double dt, double r[3][MAXPART], double v[3][MAXPART], double a[3][MAXPART]) {
    double psum = 0.;
    
    //  Update positions and velocity with current velocity and acceleration
    for (int j=0; j<3; j++) {
        for (int i=0; i<N; i++) {
            v[j][i] += 0.5*a[j][i]*dt;
            r[j][i] += v[j][i]*dt;
        }
    }
    //  Update accellerations from updated positions and compute potential energy
    double pe = computeAccelerationsAndPotential(N, r, a);

    // Elastic walls
    for (int j=0; j<3; j++) {
        for (int i=0; i<N; i++) {
            v[j][i] += 0.5*a[j][i]*dt; //  Update velocity with updated acceleration

            if (r[j][i]<0. || r[j][i]>=L) {
                v[j][i] *=-1.; //- elastic walls
                psum += 2*m*fabs(v[j][i])/dt;  // contribution to pressure from "left" walls
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


void initializeVelocities(int N, double Tinit, double v[3][MAXPART]) {
    
    int i, j;
    
    for (i=0; i<N; i++) {
        for (j=0; j<3; j++) {
            //  Pull a number from a Gaussian Distribution
            v[j][i] = gaussdist();
            
        }
    }
    
    // Vcm = sum_i^N  m*v_i/  sum_i^N  M
    // Compute center-of-mas velocity according to the formula above
    double vCM[3] = {0, 0, 0};
    
    for (j=0; j<3; j++) {
        for (i=0; i<N; i++) {
            vCM[j] += m*v[j][i];
        }
    }
    
    for (i=0; i<3; i++) vCM[i] /= N*m;
    
    //  Subtract out the center-of-mass velocity from the
    //  velocity of each particle... effectively set the
    //  center of mass velocity to zero so that the system does
    //  not drift in space!
    for (j=0; j<3; j++) {
        for (i=0; i<N; i++) {
            v[j][i] -= vCM[j];
        }
    }
    
    //  Now we want to scale the average velocity of the system
    //  by a factor which is consistent with our initial temperature, Tinit
    double vSqdSum, lambda;
    vSqdSum=0.;
    for (j=0; j<3; j++) {
        for (i=0; i<N; i++) {
            vSqdSum += v[j][i]*v[j][i];
        }
    }
    
    lambda = sqrt( 3*(N-1)*Tinit/vSqdSum);
    
    for (j=0; j<3; j++) {
        for (i=0; i<N; i++) {
            v[j][i] *= lambda;
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
