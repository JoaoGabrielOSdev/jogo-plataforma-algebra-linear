# Platform Hunter — Álgebra Linear aplicada a jogos

Jogo de plataforma 2D em Python (Pygame + NumPy) desenvolvido na disciplina **Matemática Aplicada à Computação**, com melhorias que exploram conceitos de Álgebra Linear.

## Como rodar

```bash
pip install pygame numpy
python jogo_algebra_linear.py
```

**Controles:** A/D ou setas para mover · W/Espaço para pular · R para reiniciar · ESC para sair

## Conceitos de Álgebra Linear no código original

| Conceito | Onde aparece |
|---|---|
| Vetores em ℝ² | `player.pos`, `velocity`, `enemy.pos` são `np.array` com 2 componentes |
| Base canônica | `E1 = [1, 0]` e `E2 = [0, 1]` |
| Soma de vetores e escalar | `self.pos += self.velocity * dt` e `self.velocity += gravity * dt` |
| Norma e distância | `distance = np.linalg.norm(direction)` |
| Normalização | `direction_unit = direction / distance` |
| Produto interno | `dot = np.dot(self.forward, direction_unit)`, comparado com `cos(40°)` para o campo de visão |
| Matriz de rotação | `rotation_matrix(theta)` = [[cos θ, −sin θ], [sin θ, cos θ]] |
| Transformação linear + translação | `transform_points`: `world = (R @ S) @ p + position` |

## Melhorias implementadas

Os trechos novos estão marcados com `[MELHORIA]` no código.

### 1. Projéteis guiados por vetor unitário
Quando o inimigo vê o jogador, ele atira. A velocidade do projétil é um **escalar vezes o vetor unitário** que aponta para o jogador:

```python
velocity = self.projectile_speed * direction_unit
```

Assim todo projétil tem a mesma rapidez (320 px/s), independente da distância.

### 2. Reflexão nas plataformas (produto interno)
Ao bater numa plataforma, o projétil quica usando a fórmula de reflexão:

**v' = v − 2 (v · n) n**

onde **n** é a normal unitária da face atingida (±E1 nas laterais, ±E2 no topo e no fundo). O termo `(v · n) n` é a **projeção** de v sobre n; subtraí-lo duas vezes inverte só a componente perpendicular à superfície. Exemplo: v = [3, 4] batendo no chão (n = [0, −1]) vira [3, −4].

### 3. Cone de visão desenhado com matriz de rotação
Antes o campo de visão era desenhado como um círculo. Agora as bordas do cone são o vetor `forward` **girado por ±40°**:

```python
edge = rotation_matrix(alpha) @ enemy.forward * radius
```

O cone fica vermelho quando o jogador é detectado, e o raio desenhado é o alcance real da detecção.

### 4. Varredura do olhar na patrulha
Durante a patrulha, o inimigo "olha" para cima e para baixo. O ângulo varia com o tempo e o vetor `forward` é obtido por rotação da base canônica:

θ(t) = π + 25° · sin(1,5 t),  `forward = R(θ) @ E1`

### 5. Ângulo pelo produto interno no HUD
Como `forward` e `direction_unit` são unitários, `forward · direction_unit = cos θ`. O HUD mostra o ângulo entre eles em graus:

θ = arccos(forward · direction_unit)
