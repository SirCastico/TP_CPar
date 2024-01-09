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
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>
#include <stdalign.h>
#include <stdint.h>
#include <immintrin.h>
#include <vector>

//  Lennard-Jones parameters in natural units!
const double sigma = 1.;
const double sigma12 = 1.;
const double sigma6 = 1.;
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

__global__ void updateVelocityAndPositions(int N, double r[3][MAXPART], double a[3][MAXPART], double v[3][MAXPART], double dt);
__global__ void computeAccelerationsAndPotential(int N, const double r[3][MAXPART],double a[3][MAXPART], uint16_t *block_i, double *pot_out);
__global__ void updateVelocityAndPressure(int N, double r[3][MAXPART], double v[3][MAXPART], double a[3][MAXPART], double dt, int L, double pressure[MAXPART]);
__global__ void reduceSum(double *arr_in, double *arr_out);
__global__ void sumOfLengths(int N, const double v[3][MAXPART], double* result);

__device__ double myAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

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

template<typename T>
struct DeviceBuffer{
    T *buf;
    size_t len;

    void free(){
        cudaFree(this->buf);
    }
    void memset(uint8_t byte){
        cudaMemset(this->buf, byte, this->len*sizeof(T));
    }
    void memcpyToHost(T *other){
        cudaMemcpy(other, this->buf, this->len*sizeof(T),cudaMemcpyDeviceToHost);
    }
    static DeviceBuffer newBuffer(size_t len){
        DeviceBuffer<T> db{};
        auto error = cudaMalloc(&db.buf, len*sizeof(T));
        if(error!=cudaSuccess){
            printf("cudamalloc error:%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
            exit(1);
        }
        db.len = len;
        return db;
    }

    static DeviceBuffer fromCPUBuffer(T *cpu_buf, size_t len){
        DeviceBuffer<T> db{};
        auto error = cudaMalloc(&db.buf, len*sizeof(T));
        if(error!=cudaSuccess){
            printf("cudamalloc error:%s - %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
            exit(1);
        }
        cudaMemcpy(db.buf, cpu_buf, len*sizeof(T), cudaMemcpyHostToDevice);
        db.len = len;
        return db;
    }
};


struct Reductor{
    uint32_t nThreads;
    DeviceBuffer<double> in, local;

    void reduce(double *out){
        uint32_t nBlocksReduce = local.len;
        DeviceBuffer<double> swap{};
        while(nBlocksReduce>nThreads*2){
            reduceSum<<<nBlocksReduce, nThreads, nThreads*sizeof(double)>>>(in.buf, local.buf);
            swap = in;
            in = local;
            local = swap;
            nBlocksReduce = ceil(in.len/nThreads*2);
            if(nBlocksReduce<=nThreads*2){
                reduceSum<<<1,nBlocksReduce, nThreads*sizeof(double)>>>(in.buf, out);
                break;
            }
        }
    }

    void free(){
        in.free();
        local.free();
    }

    static Reductor newReductor(uint32_t len, uint32_t nThreads){
        Reductor red;
        red.in = DeviceBuffer<double>::newBuffer(len);
        red.local = DeviceBuffer<double>::newBuffer(ceil((float)len/(nThreads*2)));
        red.nThreads = nThreads;
        return red;
    }
};

uint32_t combinations(uint32_t n, uint32_t c){
    uint32_t x = n;
    for(int i=1;i<c;i++){
        x*=n-i;
    }
    uint32_t d = c;
    for(int i=1;i<c-1;i++){
        d*=c-i;
    }
    return x/d;
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
    int N = 5000;

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

    double gc, Z;

    int nThreads = 64;
    int nBlocks = ceil((float)N / (float)nThreads);

    int combs = combinations(N,2);
    int nBlocksComb = ceil((float)combs/(float)nThreads);

    printf("allocating dev rav buffers\n");
    auto dev_r = DeviceBuffer<double[3][MAXPART]>::fromCPUBuffer(&r, 1);
    auto dev_a = DeviceBuffer<double[3][MAXPART]>::fromCPUBuffer(&a, 1);
    auto dev_v = DeviceBuffer<double[3][MAXPART]>::fromCPUBuffer(&v, 1);

    printf("allocating dev sim val buffers\n");
    auto dev_pressure = DeviceBuffer<double>::newBuffer(NumTime*nThreads*2);
    auto dev_lsum = DeviceBuffer<double>::newBuffer(NumTime*nThreads*2);
    auto dev_pot = DeviceBuffer<double>::newBuffer(NumTime*nThreads*2);
    dev_pressure.memset(0);
    dev_lsum.memset(0);
    dev_pot.memset(0);

    printf("calculating block_i n:%d\n", nBlocksComb);
    std::vector<uint16_t> cpu_block_i{};
    cpu_block_i.reserve(nBlocksComb);
    for(uint16_t i=0,j=0;;){
        j+=nThreads*2;
        if(j>=N){
            i++;
            j=i+1;
            if(j>=N)
                break;
        }
        cpu_block_i.push_back(i);
    }

    printf("allocating block_i\n");
    auto block_i = DeviceBuffer<uint16_t>::fromCPUBuffer(cpu_block_i.data(),nBlocksComb);

    printf("allocating reductors\n");
    auto pot_red = Reductor::newReductor(nBlocksComb, nThreads);
    auto lsum_red = Reductor::newReductor(nBlocks, nThreads);
    auto pr_red = Reductor::newReductor(nBlocks, nThreads);

    printf("starting simulation loop\n");
    for (int i=0; i <= NumTime; i++) {

        updateVelocityAndPositions<<< nBlocks, nThreads >>>(N, *dev_r.buf, *dev_v.buf, *dev_a.buf, dt);
        auto error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("updVelPos");
        dev_a.memset(0);
        computeAccelerationsAndPotential<<<nBlocksComb,nThreads>>>(N, *dev_a.buf, *dev_a.buf, block_i.buf, pot_red.in.buf);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("AccelPot");
        pot_red.reduce(dev_pot.buf+i*nThreads*2);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("AccelPotReduce");

        updateVelocityAndPressure<<<nBlocks,nThreads>>>(N, *dev_r.buf, *dev_v.buf, *dev_a.buf, dt, L, pr_red.in.buf);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("velPress");
        pr_red.reduce(dev_pressure.buf+i*nThreads*2);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("VelPressRed");

        sumOfLengths<<<nBlocks,nThreads>>>(N, *dev_v.buf, lsum_red.in.buf);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("sumLenghts");
        lsum_red.reduce(dev_lsum.buf+i*nThreads*2);
        error = cudaDeviceSynchronize();
        if(error!=cudaSuccess)
            throw std::runtime_error("sumLenReduce");
    }

    printf("allocating dev out\n");
    auto dev_pressure_out = DeviceBuffer<double>::newBuffer(NumTime);
    auto dev_lsum_out = DeviceBuffer<double>::newBuffer(NumTime);
    auto dev_pot_out = DeviceBuffer<double>::newBuffer(NumTime);
    int nBlocksOut = ceil((float)dev_pressure.len/(float)nThreads*2);
    printf("reducing out\n");
    reduceSum<<<nBlocksOut,nThreads,nBlocksOut*sizeof(double)>>>(dev_pressure.buf, dev_pressure_out.buf);
    reduceSum<<<nBlocksOut,nThreads,nBlocksOut*sizeof(double)>>>(dev_lsum.buf,dev_lsum_out.buf);
    reduceSum<<<nBlocksOut,nThreads,nBlocksOut*sizeof(double)>>>(dev_pot.buf,dev_pot_out.buf);

    double host_pressure_out[NumTime];
    double host_lsum_out[NumTime];
    double host_pot_out[NumTime];

    printf("copying out\n");
    dev_pressure_out.memcpyToHost(host_pressure_out);
    dev_lsum_out.memcpyToHost(host_lsum_out);
    dev_pot_out.memcpyToHost(host_pot_out);

    for(int i=0;i<=NumTime;++i){
        double time = i * dt * timefac;
        double pressure = host_pressure_out[i]*PressFac;
        double msv = host_lsum_out[i]/N;
        double ke = m*host_lsum_out[i]/2.0;
        double temp = m*msv/(3*kB) * TempFac;
        double pe = host_pot_out[i];

        Tavg += temp;
        Pavg += pressure;
        
        fprintf(ofp, "  %8.4e  %20.8f  %20.8f %20.8f  %20.8f  %20.8f \n",
            time, temp, pressure, ke, pe, ke+pe);
    }
    
    dev_r.free();
    dev_a.free();
    dev_v.free();

    dev_pressure.free();
    dev_lsum.free();
    dev_pot.free();

    block_i.free();

    pot_red.free();
    lsum_red.free();
    pr_red.free();

    dev_pressure.free();
    dev_lsum.free();
    dev_pot.free();

    dev_pressure_out.free();
    dev_lsum_out.free();
    dev_pot_out.free();
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
        
    double v20=0.,v21=0.,v22 = 0.;
    for (int i=0; i<N; i++) {
        v20 += v[0][i]*v[0][i];
        v21 += v[1][i]*v[1][i];
        v22 += v[2][i]*v[2][i];
    }
    double kin = m*(v20+v21+v22)/2.;
    
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

__global__ void updateVelocityAndPositions(int N, double r[3][MAXPART], double a[3][MAXPART], double v[3][MAXPART], double dt){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=N) return;

    v[0][id] += 0.5*a[0][id]*dt;
    v[1][id] += 0.5*a[1][id]*dt;
    v[2][id] += 0.5*a[2][id]*dt;

    r[0][id] += v[0][id]*dt;
    r[1][id] += v[1][id]*dt;
    r[2][id] += v[2][id]*dt;
}

__global__ void updateVelocityAndPressure(int N, double r[3][MAXPART], double v[3][MAXPART], double a[3][MAXPART], double dt, int L, double pressure[MAXPART]){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id>=N) return;

    v[0][id] += 0.5*a[0][id]*dt;
    v[1][id] += 0.5*a[1][id]*dt;
    v[2][id] += 0.5*a[2][id]*dt;

    int b0 = r[0][id]<0. || r[0][id]>=L;
    v[0][id] *= (b0*-2+1);
    pressure[id] = b0*2*m*fabs(v[0][id])/dt;

    int b1 = r[1][id]<0. || r[1][id]>=L;
    v[1][id] *= (b1*-2+1);
    pressure[id] = b1*2*m*fabs(v[1][id])/dt;

    int b2 = r[2][id]<0. || r[2][id]>=L;
    v[2][id] *= (b2*-2+1);
    pressure[id] = b2*2*m*fabs(v[2][id])/dt;
}


// reduce sum arr_in[blockIdx.x * blockDim.x] to arr_out[blockIdx.x]
__global__ void reduceSum(double *arr_in, double *arr_out){
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    extern __shared__ double s_arr[];

    s_arr[tid] = arr_in[i] + arr_in[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_arr[tid] += s_arr[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        arr_out[blockIdx.x] = s_arr[0];
    }
}

__global__ void sumOfLengths(int N, const double v[3][MAXPART], double* result) {
    unsigned int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sdata[];
    sdata[tid] = 0.0;

    if (i < N) {
        double v20 = v[0][i] * v[0][i];
        double v21 = v[1][i] * v[1][i];
        double v22 = v[2][i] * v[2][i];
        sdata[tid] = v20 + v21 + v22;
    }

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

__global__ void computeAccelerationsAndPotential(int N, const double r[3][MAXPART],double a[3][MAXPART], uint16_t *block_i, double *pot_out){
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ double sdata[];
    sdata[tid] = 0;
    __syncthreads();

    uint32_t i = block_i[bid];
    uint32_t j = bid * blockDim.x + tid + i + 1;
    if(j>=N){
        i+=1;
        j=i+1;
    }
    double rij[3];
    if(i<N-1){
        rij[0] = r[0][i] - r[0][j];
        rij[1] = r[1][i] - r[1][j];
        rij[2] = r[2][i] - r[2][j];

        double rSqd = (rij[0] * rij[0])+(rij[1] * rij[1])+(rij[2] * rij[2]);
        double r2p3 = rSqd*rSqd*rSqd;
        sdata[tid] = 2*4*epsilon*(sigma12/r2p3 - sigma6)/r2p3;

        double f = (48 - 24*r2p3) / rSqd*rSqd*rSqd*rSqd*rSqd*rSqd*rSqd;

        rij[0] *= f; 
        rij[1] *= f; 
        rij[2] *= f; 

        myAtomicAdd(&a[0][i], rij[0]);
        myAtomicAdd(&a[1][i], rij[1]);
        myAtomicAdd(&a[2][i], rij[2]);

        myAtomicAdd(&a[0][j], -rij[0]);
        myAtomicAdd(&a[1][j], -rij[1]);
        myAtomicAdd(&a[2][j], -rij[2]);
    }


    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if(tid==0) pot_out[bid] = sdata[0];
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
