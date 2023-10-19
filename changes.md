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

## change 6

                 0      context-switches:u               #    0.000 /sec                      
                 0      cpu-migrations:u                 #    0.000 /sec                      
               119      page-faults:u                    #   15.848 /sec                      
    17,358,838,677      cycles:u                         #    2.312 GHz                         (83.34%)
         2,671,750      stalled-cycles-frontend:u        #    0.02% frontend cycles idle        (83.34%)
    14,800,237,409      stalled-cycles-backend:u         #   85.26% backend cycles idle         (83.34%)
    23,000,123,012      instructions:u                   #    1.32  insn per cycle            
                                                  #    0.64  stalled cycles per insn     (83.34%)
       595,609,282      branches:u                       #   79.322 M/sec                       (83.33%)
         1,245,894      branch-misses:u                  #    0.21% of all branches             (83.31%)

### Função Potential
- cortadas algumas iterações dos loops

## change 7

                 0      context-switches:u               #    0.000 /sec                      
                 0      cpu-migrations:u                 #    0.000 /sec                      
               117      page-faults:u                    #   18.041 /sec                      
    14,985,958,482      cycles:u                         #    2.311 GHz                         (83.31%)
         2,073,283      stalled-cycles-frontend:u        #    0.01% frontend cycles idle        (83.35%)
    11,916,161,209      stalled-cycles-backend:u         #   79.52% backend cycles idle         (83.34%)
    26,406,705,992      instructions:u                   #    1.76  insn per cycle            
                                                  #    0.45  stalled cycles per insn     (83.34%)
       476,264,105      branches:u                       #   73.438 M/sec                       (83.30%)
           464,566      branch-misses:u                  #    0.10% of all branches             (83.35%)

- Potential e computeAccelerations unidas