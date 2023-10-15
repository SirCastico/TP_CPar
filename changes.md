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