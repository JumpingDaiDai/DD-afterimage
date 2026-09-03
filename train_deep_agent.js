// 深度強化訓練引擎：AFTERIMAGE 遺傳演算法大規模多世代演化訓練器
// 特性：500 世代深度演化、三階段難度模型輸出 (Apprentice / Assassin / ChronoLord)

const fs = require('fs');

class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize, weights = null) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    if (weights) {
      this.w1 = weights.w1;
      this.b1 = weights.b1;
      this.w2 = weights.w2;
      this.b2 = weights.b2;
    } else {
      this.w1 = Array.from({ length: inputSize }, () =>
        Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 1.5)
      );
      this.b1 = Array.from({ length: hiddenSize }, () => (Math.random() - 0.5) * 0.5);
      this.w2 = Array.from({ length: hiddenSize }, () =>
        Array.from({ length: outputSize }, () => (Math.random() - 0.5) * 1.5)
      );
      this.b2 = Array.from({ length: outputSize }, () => (Math.random() - 0.5) * 0.5);
    }
  }

  forward(inputs) {
    const hidden = new Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputs[i] * brainMul(inputs[i], this.w1[i][j]);
      }
      hidden[j] = Math.tanh(sum);
    }
    const outputs = new Array(this.outputSize);
    for (let k = 0; k < this.outputSize; k++) {
      let sum = this.b2[k];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += hidden[j] * this.w2[j][k];
      }
      outputs[k] = Math.tanh(sum);
    }
    return outputs;
  }

  mutate(rate = 0.08, scale = 0.25) {
    const copyWeights = {
      w1: this.w1.map(row => row.map(v => Math.random() < rate ? v + (Math.random() - 0.5) * scale : v)),
      b1: this.b1.map(v => Math.random() < rate ? v + (Math.random() - 0.5) * scale : v),
      w2: this.w2.map(row => row.map(v => Math.random() < rate ? v + (Math.random() - 0.5) * scale : v)),
      b2: this.b2.map(v => Math.random() < rate ? v + (Math.random() - 0.5) * scale : v)
    };
    return new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize, copyWeights);
  }

  toJSON() {
    return { w1: this.w1, b1: this.b1, w2: this.w2, b2: this.b2 };
  }
}

function brainMul(a, b) { return a * b; }

function pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
  if (l2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
}

function simulateMatch(agentA, agentB, maxFrames = 1800) {
  const state = {
    pA: {
      x: 200, y: 300, radius: 10, speed: 2.8, hp: 100,
      facingAngle: 0, attackCd: 0, attackSpeed: 16,
      echoCharges: 2, history: [], maxHistory: 180, resonanceDmg: 0.9,
      invulnerable: 0, score: 0
    },
    pB: {
      x: 600, y: 300, radius: 10, speed: 2.8, hp: 100,
      facingAngle: Math.PI, attackCd: 0, attackSpeed: 16,
      echoCharges: 2, history: [], maxHistory: 180, resonanceDmg: 0.9,
      invulnerable: 0, score: 0
    },
    echoesA: [], echoesB: [],
    bulletsA: [], bulletsB: []
  };

  const W = 800, H = 600;

  for (let f = 0; f < maxFrames; f++) {
    if (state.pA.hp <= 0 || state.pB.hp <= 0) break;

    // Agent A 輸入 (8 維)
    const inA = [
      (state.pB.x - state.pA.x) / W,
      (state.pB.y - state.pA.y) / H,
      Math.hypot(state.pB.x - state.pA.x, state.pB.y - state.pA.y) / 1000,
      Math.cos(state.pA.facingAngle - state.pB.facingAngle),
      state.echoesB.length / 2,
      state.echoesB.length > 0 ? pointToSegmentDist(state.pA.x, state.pA.y, state.pB.x, state.pB.y, state.echoesB[0].x, state.echoesB[0].y) / 500 : 1.0,
      state.pA.echoCharges / 2,
      state.pA.hp / 100
    ];
    const outA = agentA.forward(inA);

    // Agent B 輸入 (8 維)
    const inB = [
      (state.pA.x - state.pB.x) / W,
      (state.pA.y - state.pB.y) / H,
      Math.hypot(state.pA.x - state.pB.x, state.pA.y - state.pB.y) / 1000,
      Math.cos(state.pB.facingAngle - state.pA.facingAngle),
      state.echoesA.length / 2,
      state.echoesA.length > 0 ? pointToSegmentDist(state.pB.x, state.pB.y, state.pA.x, state.pA.y, state.echoesA[0].x, state.echoesA[0].y) / 500 : 1.0,
      state.pB.echoCharges / 2,
      state.pB.hp / 100
    ];
    const outB = agentB.forward(inB);

    // A 移動
    const moveLenA = Math.hypot(outA[0], outA[1]);
    if (moveLenA > 0.1) {
      state.pA.x += (outA[0] / moveLenA) * state.pA.speed;
      state.pA.y += (outA[1] / moveLenA) * state.pA.speed;
      state.pA.facingAngle = Math.atan2(outA[1], outA[0]);
    }
    state.pA.x = Math.max(15, Math.min(W - 15, state.pA.x));
    state.pA.y = Math.max(15, Math.min(H - 15, state.pA.y));
    state.pA.history.push({ x: state.pA.x, y: state.pA.y });
    if (state.pA.history.length > state.pA.maxHistory) state.pA.history.shift();

    if (outA[2] > 0.65 && state.pA.echoCharges > 0 && state.pA.history.length >= 15) {
      state.pA.echoCharges--;
      const oldState = state.pA.history[0];
      state.echoesA.push({ x: state.pA.x, y: state.pA.y, life: 360 });
      state.pA.x = oldState.x;
      state.pA.y = oldState.y;
      state.pA.invulnerable = 30;
      state.pA.score += 60; // 戰術回溯加分
    }

    // B 移動
    const moveLenB = Math.hypot(outB[0], outB[1]);
    if (moveLenB > 0.1) {
      state.pB.x += (outB[0] / moveLenB) * state.pB.speed;
      state.pB.y += (outB[1] / moveLenB) * state.pB.speed;
      state.pB.facingAngle = Math.atan2(outB[1], outB[0]);
    }
    state.pB.x = Math.max(15, Math.min(W - 15, state.pB.x));
    state.pB.y = Math.max(15, Math.min(H - 15, state.pB.y));
    state.pB.history.push({ x: state.pB.x, y: state.pB.y });
    if (state.pB.history.length > state.pB.maxHistory) state.pB.history.shift();

    if (outB[2] > 0.65 && state.pB.echoCharges > 0 && state.pB.history.length >= 15) {
      state.pB.echoCharges--;
      const oldState = state.pB.history[0];
      state.echoesB.push({ x: state.pB.x, y: state.pB.y, life: 360 });
      state.pB.x = oldState.x;
      state.pB.y = oldState.y;
      state.pB.invulnerable = 30;
      state.pB.score += 60;
    }

    // 開火
    state.pA.attackCd--;
    if (state.pA.attackCd <= 0) {
      state.bulletsA.push({ x: state.pA.x, y: state.pA.y, vx: Math.cos(state.pA.facingAngle) * 9.5, vy: Math.sin(state.pA.facingAngle) * 9.5, damage: 20, life: 50 });
      state.pA.attackCd = state.pA.attackSpeed;
    }
    state.pB.attackCd--;
    if (state.pB.attackCd <= 0) {
      state.bulletsB.push({ x: state.pB.x, y: state.pB.y, vx: Math.cos(state.pB.facingAngle) * 9.5, vy: Math.sin(state.pB.facingAngle) * 9.5, damage: 20, life: 50 });
      state.pB.attackCd = state.pB.attackSpeed;
    }

    // 子彈碰撞判定
    for (let i = state.bulletsA.length - 1; i >= 0; i--) {
      const b = state.bulletsA[i];
      b.x += b.vx; b.y += b.vy; b.life--;
      if (Math.hypot(b.x - state.pB.x, b.y - state.pB.y) < state.pB.radius + 3) {
        if (state.pB.invulnerable <= 0) {
          state.pB.hp -= b.damage;
          state.pA.score += 180;
          state.pB.score -= 90;
        }
        state.bulletsA.splice(i, 1);
        continue;
      }
      if (b.life <= 0) state.bulletsA.splice(i, 1);
    }
    for (let i = state.bulletsB.length - 1; i >= 0; i--) {
      const b = state.bulletsB[i];
      b.x += b.vx; b.y += b.vy; b.life--;
      if (Math.hypot(b.x - state.pA.x, b.y - state.pA.y) < state.pA.radius + 3) {
        if (state.pA.invulnerable <= 0) {
          state.pA.hp -= b.damage;
          state.pB.score += 180;
          state.pA.score -= 90;
        }
        state.bulletsB.splice(i, 1);
        continue;
      }
      if (b.life <= 0) state.bulletsB.splice(i, 1);
    }

    // 雷射連線切割判定
    if (state.echoesA.length > 0) {
      const ec = state.echoesA[0];
      if (pointToSegmentDist(state.pB.x, state.pB.y, state.pA.x, state.pA.y, ec.x, ec.y) < state.pB.radius + 4) {
        state.pB.hp -= state.pA.resonanceDmg;
        state.pA.score += 20; // 雷射切割加分
      }
    }
    if (state.echoesB.length > 0) {
      const ec = state.echoesB[0];
      if (pointToSegmentDist(state.pA.x, state.pA.y, state.pB.x, state.pB.y, ec.x, ec.y) < state.pA.radius + 4) {
        state.pA.hp -= state.pB.resonanceDmg;
        state.pB.score += 20;
      }
    }

    if (state.pA.invulnerable > 0) state.pA.invulnerable--;
    if (state.pB.invulnerable > 0) state.pB.invulnerable--;
  }

  if (state.pA.hp > 0 && state.pB.hp <= 0) state.pA.score += 1500;
  if (state.pB.hp > 0 && state.pA.hp <= 0) state.pB.score += 1500;

  return { scoreA: state.pA.score, scoreB: state.pB.score };
}

// 啟動 500 世代深度訓練
console.log("==================================================");
console.log("啟動 AFTERIMAGE 大規模神經網絡深度訓練器 (500 Generations)");
console.log("==================================================");

const POPULATION_SIZE = 50;
const GENERATIONS = 500;
const INPUT_SIZE = 8;
const HIDDEN_SIZE = 12;
const OUTPUT_SIZE = 4;

let population = Array.from({ length: POPULATION_SIZE }, () => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));

let modelApprentice = null; // 第 50 代
let modelAssassin = null;   // 第 200 代
let modelChronoLord = null; // 第 500 代

for (let gen = 1; gen <= GENERATIONS; gen++) {
  const scores = new Array(POPULATION_SIZE).fill(0);

  for (let i = 0; i < POPULATION_SIZE; i++) {
    for (let j = 0; j < 4; j++) {
      const oppIndex = (i + 1 + j) % POPULATION_SIZE;
      const res = simulateMatch(population[i], population[oppIndex]);
      scores[i] += res.scoreA;
      scores[oppIndex] += res.scoreB;
    }
  }

  const ranked = population.map((net, idx) => ({ net, score: scores[idx] }))
    .sort((a, b) => b.score - a.score);

  if (gen % 50 === 0 || gen === 1) {
    const avg = Math.round(scores.reduce((a, b) => a + b) / POPULATION_SIZE);
    console.log(`[世代 Gen ${String(gen).padStart(3, '0')}/${GENERATIONS}] 最高適應度: ${String(Math.round(ranked[0].score)).padStart(6, ' ')} | 平均適應度: ${String(avg).padStart(6, ' ')}`);
  }

  if (gen === 50) modelApprentice = ranked[0].net.toJSON();
  if (gen === 200) modelAssassin = ranked[0].net.toJSON();
  if (gen === GENERATIONS) modelChronoLord = ranked[0].net.toJSON();

  // 保留前 20% 精英，突變衍生下一代
  const newPop = [];
  const eliteCount = Math.floor(POPULATION_SIZE * 0.2);
  for (let i = 0; i < eliteCount; i++) {
    newPop.push(ranked[i].net);
  }
  while (newPop.length < POPULATION_SIZE) {
    const parent = ranked[Math.floor(Math.random() * eliteCount)].net;
    newPop.push(parent.mutate(0.08, 0.25));
  }
  population = newPop;
}

// 輸出三階難度模型權重集合
const modelsBundle = {
  apprentice: modelApprentice,
  assassin: modelAssassin,
  chronoLord: modelChronoLord
};

fs.writeFileSync('/Users/jason/AI Agent/Games/dd-afterimage/trained_models_bundle.json', JSON.stringify(modelsBundle, null, 2));
console.log("==================================================");
console.log("訓練圓滿完成！已成功導出三階難度模型集合：trained_models_bundle.json");
console.log("1. 時空學徒 (Apprentice - Gen 50)");
console.log("2. 幾何刺客 (Assassin - Gen 200)");
console.log("3. 時間領主 (Chrono Lord - Gen 500)");
console.log("==================================================");
