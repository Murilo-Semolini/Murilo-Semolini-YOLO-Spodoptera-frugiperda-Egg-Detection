"# YOLO-Ovos-De-Lagarta"
Devido a alguns problemas, as classes ficaram um pouco bagunçadas no meu dataset
Classe de index 0 = 2 ovos (nomeada de "2_Eggs")
Classe de index 1 = 3 ovos (nomeada de "3_Eggs")
Classe de index 2 = 1 ovo (nomeada de "1_Egg")

Neste repositório existem códigos para o teste das redes neurais

ComparacaoDeModelos.py serve para testar os 4 modelos ao mesmo tempo e verificar os resultados encontrados por cada um vs as anotações de um humano

contagem_vs_predicao.py é similar, mas testa apenas um único modelo

teste1Modelo1Imagem serve para que um modelo analise uma única imagem que não está anotada e diga quanto ovos foram achados

teste1ModeloVariasImagens serve para fazer a mesma análise, porém em várias imagens

mainCode.py é o código usado para configurar e realizar o treinamento dos modelos
