# Ridge e LASSO como casos especiais do Elastic Net

Questão 3(a).

## A penalidade

O Elastic Net estima os coeficientes minimizando

$$
\frac{1}{2n}\sum_{i=1}^{n}\left(y_i - \beta_0 - x_i^\top\beta\right)^2
\;+\; \lambda \left[ \frac{1-\alpha}{2} \sum_{j=1}^{p} \beta_j^2 \;+\; \alpha \sum_{j=1}^{p} |\beta_j| \right],
\qquad \lambda \ge 0,\; 0 \le \alpha \le 1.
$$

O primeiro termo é o ajuste aos dados. O segundo é a penalidade, que mistura uma parte $L_2$ (quadrática) e uma parte $L_1$ (valor absoluto) com peso $\alpha$.

## Caso $\alpha = 0$: Ridge

$$
\lambda \left[ \frac{1-0}{2} \sum_j \beta_j^2 + 0 \cdot \sum_j |\beta_j| \right]
= \frac{\lambda}{2} \sum_j \beta_j^2 .
$$

Sobra apenas a penalidade quadrática $\|\beta\|_2^2$, que é a do Ridge. O fator $1/2$ só muda a escala de $\lambda$: o Ridge com parâmetro $\lambda' = \lambda/2$ tem exatamente a mesma solução.

## Caso $\alpha = 1$: LASSO

$$
\lambda \left[ \frac{1-1}{2} \sum_j \beta_j^2 + 1 \cdot \sum_j |\beta_j| \right]
= \lambda \sum_j |\beta_j| .
$$

Sobra apenas a penalidade $\|\beta\|_1$, que é a do LASSO.

## Conclusão

Ridge ($\alpha = 0$) e LASSO ($\alpha = 1$) são os dois extremos da família Elastic Net. Valores intermediários de $\alpha$ combinam a seleção de variáveis do LASSO (a parte $L_1$ pode zerar coeficientes) com a estabilidade do Ridge diante de preditores correlacionados (a parte $L_2$ divide o peso entre eles).

## Por que isso importa no código

No scikit-learn, o parâmetro $\lambda$ se chama `alpha` e o parâmetro $\alpha$ se chama `l1_ratio`. `ElasticNet` e `Lasso` usam a função objetivo acima. A classe `Ridge`, porém, minimiza

$$
\sum_i (y_i - \beta_0 - x_i^\top\beta)^2 + a \,\|\beta\|_2^2 ,
$$

sem o fator $1/(2n)$. Multiplicando a função do Elastic Net com $\alpha = 0$ por $2n$:

$$
\sum_i (y_i - \beta_0 - x_i^\top\beta)^2 + n\lambda\,\|\beta\|_2^2
\quad\Longrightarrow\quad a = n\,\lambda .
$$

Portanto, `Ridge(alpha=λ)` **não** é o Ridge "com o mesmo λ" do LASSO e do Elastic Net: ele é $n$ vezes menos penalizado (543 vezes, no treino deste projeto). O projeto usa `Ridge(alpha=n * λ)` (função `ridge_from_lambda` em `src/models.py`), e o teste `test_ridge_equivale_elastic_net_com_alpha_zero` confirma numericamente a equivalência.
