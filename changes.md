## change 1
- refactor

## change 2
- vars locais
- restrict nas funções que recebem arrays
- memset em vez de for loop na computeAccelerations

## change 3

### Função Potential
- contas simplificadas, removido sqrt e substituidos os pow

### Função computeAccelerations
- substituidos os pow

### Função VelocityVerlet
- movido update das velocidades para o loop que calcula a pressão

### considerações

#### mudança dos vetores posição, aceleração e velocidade para uma struct, operar num array dessa struct
- Contra argumento: Potential e computeAccelerations acedem a um array de cada vez

## change 4
- movido fprintf da main loop para fora, resultados guardados num array (pouca diferença)

### Função Potential
- calculo de constantes sigma^12 e sigma^6 pré-calculado

## change 5

                 0      context-switches:u               #    0.000 /sec                      
                 0      cpu-migrations:u                 #    0.000 /sec                      
               117      page-faults:u                    #   17.147 /sec                      
    23,418,659,892      cycles:u                         #    3.432 GHz                       
         2,880,656      stalled-cycles-frontend:u        #    0.01% frontend cycles idle      
    19,004,749,245      stalled-cycles-backend:u         #   81.15% backend cycles idle       
    41,844,016,276      instructions:u                   #    1.79  insn per cycle            
                                                  #    0.45  stalled cycles per insn   
     2,352,681,177      branches:u                       #  344.793 M/sec                     
         1,333,615      branch-misses:u                  #    0.06% of all branches 

### Função Potential
- math ligeiramente simplificada

### Função computeAccelerations
- math ligeiramente simplificada
