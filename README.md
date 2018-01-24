Passos do SWT:

Use OpenCV para extrair bordas de imagem usando a detecção de borda Canny

Calcule os derivados x e y da imagem, que podem ser sobrepostos para calcular o gradiente da imagem. O gradiente descreve, para cada pixel, a direção do maior contraste. No caso de um pixel de borda, isso é sinônimo do vetor normal para a borda.

Para cada pixel de borda, percorra a direção θ do gradiente até o próximo pino de borda ser encontrado (ou você cai da imagem). Se o gradiente de pixel de borda correspondente estiver apontado na direção oposta (θ - π), sabemos que a borda recém-encontrada é aproximadamente paralela à primeira, e acabamos de cortar uma fatia (linha) através de um traçado. Registre a largura do traço, em pixels e atribua esse valor a todos os pixels na fatia que acabamos de atravessar.

Para pixels que podem pertencer a várias linhas, concilie as diferenças nessas larguras de linha, atribuindo a todos os pixels o valor médio da largura do traço. Isso permite que os dois traços que você possa encontrar em uma forma 'L' sejam considerados com a mesma largura de traçado comum.

Conecte linhas que se sobrepõem usando uma estrutura de dados union-find (disjoint-set), resultando em um conjunto disjugal de todas as fatias de ataque sobrepostas. Cada conjunto de linhas é provavelmente uma única letra / caractere.

Aplique uma filtragem inteligente para os conjuntos de linhas; devemos eliminar qualquer coisa a pequena (largura, altura) para ser um personagem legível, bem como qualquer coisa muito longa ou gorda (largura: proporção de altura), ou muito esparsa (diâmetro: proporção da largura do traço) para ser realisticamente um personagem.

Use uma árvore kd para encontrar emparelhamentos de formas de acasticamento semelhante (com base na largura do traçado) e cruze isso com emparelhamentos de formas de tamanho similar (com base em altura / largura). Calcule o ângulo do texto entre esses dois caracteres (inclinados para cima, para baixo?).

Use uma árvore kd para encontrar emparelhamentos de correspondências de letras com orientações semelhantes. Esses agrupamentos de letras provavelmente formam uma palavra. Encadear pares semelhantes juntos.

Produza uma imagem final contendo as palavras resultantes.