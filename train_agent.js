// 訓練腳本：AFTERIMAGE 遺傳演算法離線神經網絡訓練器 (Headless Evolutionary Trainer)
// 零外部依賴、純 JS 矩陣計算、1000x 無渲染極速模擬

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
      // 隨機初始化高斯權重
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
    // 隱藏層 (Tanh 激活函數)
    const hidden = new Array(this.hiddenSize);
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < this.inputSize; i++) {
        sum += inputs[i] * this.w1[i][j];
      }
      hidden[j] = Math.tanh(sum);
    }

    // 輸出層 (Tanh 激活函數)
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

  mutate(mutationRate = 0.08, mutationScale = 0.25) {
    const copyWeights = {
      w1: this.w1.map(row => row.map(v => Math.random() < mutationRate ? v + (Math.random() - 0.5) * mutationScale : v)),
      b1: this.b1.map(v => Math.random() < mutationRate ? v + (Math.random() - 0.5) * mutationScale : v),
      w2: this.w2.map(row => row.map(v => Math.random() < mutationRate ? v + (Math.random() - 0.5) * mutationScale : v)),
      b2: this.b2.map(v => Math.random() < mutationRate ? v + (Math.random() - 0.5) * mutationScale : v)
    };
    return new NeuralNetwork(this.inputSize, this.hiddenSize, this.outputSize, copyWeights);
  }

  toJSON() {
    return { w1: this.w1, b1: this.b1, w2: this.w2, b2: this.b2 };
  }
}

// 幾何工具
function pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const l2 = (x2 - x1) ** 2 + (y2 - y1) ** 2;
  if (l2 === 0) return Math.hypot(px - x1, py - y1);
  let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * (x2 - x1)), py - (y1 + t * (y2 - y1)));
}

// 快速對局模擬 (模擬 2 個 AI 互相 1v1 對抗)
function simulateMatch(agentA, agentB, maxFrames = 1800) {
  const state = {
    pA: {
      x: 200, y: 300, radius: 10, speed: 2.8, hp: 100, maxHp: 100,
      facingAngle: 0, attackCd: 0, attackSpeed: 16,
      echoCharges: 2, maxEchoCharges: 2, echoChargeTimer: 0, maxEchoChargeCd: 150,
      history: [], maxHistory: 180, resonanceDmg: 0.9,
      invulnerable: 0, score: 0
    },
    pB: {
      x: 600, y: 300, radius: 10, speed: 2.8, hp: 100, maxHp: 100,
      facingAngle: Math.PI, attackCd: 0, attackSpeed: 16,
      echoCharges: 2, maxEchoCharges: 2, echoChargeTimer: 0, maxEchoChargeCd: 150,
      history: [], maxHistory: 180, resonanceDmg: 0.9,
      invulnerable: 0, score: 0
    },
    echoesA: [], echoesB: [],
    bulletsA: [], bulletsB: []
  };

  const W = 800, H = 600;

  for (let f = 0; f < maxFrames; f++) {
    if (state.pA.hp <= 0 || state.pB.hp <= 0) break;

    // 1. Agent A 決策
    // 輸入特徵 (8維): [相對X, 相對Y, 相對距離, 面向差, 對手殘像數, 對手雷射距離, 自身充能, 自身血量]
    const dxA = (state.pB.x - state.pA.x) / W;
    const dyA = (state.pB.y - state.pA.y) / H;
    const distA = Math.hypot(state.pB.x - state.pA.x, state.pB.y - state.pA.y) / 1000;
    let laserDistA = 1.0;
    if (state.echoesB.length > 0) {
      const ec = state.echoesB[0];
      laserDistA = pointToSegmentDist(state.pA.x, state.pA.y, state.pB.x, state.pB.y, ec.x, ec.y) / 500;
    }

    const inA = [dxA, dyA, distA, Math.cos(state.pA.facingAngle - state.pB.facingAngle), state.echoesB.length / 2, laserDistA, state.pA.echoCharges / 2, state.pA.hp / 100];
    const outA = agentA.forward(inA); // [moveDirX, moveDirY, triggerEcho (>0.6), triggerSwap (>0.6)]

    // 2. Agent B 決策
    const dxB = (state.pA.x - state.pB.x) / W;
    const dyB = (state.pA.y - state.pB.y) / H;
    const distB = Math.hypot(state.pA.x - state.pB.x, state.pA.y - state.pB.y) / 1000;
    let laserDistB = 1.0;
    if (state.echoesA.length > 0) {
      const ec = state.echoesA[0];
      laserDistB = pointToSegmentDist(state.pB.x, state.pB.y, state.pA.x, state.pA.y, ec.x, ec.y) / 500;
    }
    const inB = [dxB, dyB, distB, Math.cos(state.pB.facingAngle - state.pA.facingAngle), state.echoesA.length / 2, laserDistB, state.pB.echoCharges / 2, state.pB.hp / 100];
    const outB = agentB.forward(inB);

    // 處理移動與回溯 (A)
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
      state.echoesA.push({ frames: [...state.pA.history], currentFrame: 0, x: state.pA.x, y: state.pA.y, life: 360 });
      state.pA.x = oldState.x;
      state.pA.y = oldState.y;
      state.pA.invulnerable = 30;
      state.pA.score += 50; // 成功啟動時空戰術加分
    }

    // 處理移動與回溯 (B)
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
      state.echoesB.push({ frames: [...state.pB.history], currentFrame: 0, x: state.pB.x, y: state.pB.y, life: 360 });
      state.pB.x = oldState.x;
      state.pB.y = oldState.y;
      state.pB.invulnerable = 30;
      state.pB.score += 50;
    }

    // 主武器開火 (A)
    state.pA.attackCd--;
    if (state.pA.attackCd <= 0) {
      state.bulletsA.push({
        x: state.pA.x, y: state.pA.y,
        vx: Math.cos(state.pA.facingAngle) * 9.5, vy: Math.sin(state.pA.facingAngle) * 9.5,
        damage: 20, life: 50
      });
      state.pA.attackCd = state.pA.attackSpeed;
    }

    // 主武器開火 (B)
    state.pB.attackCd--;
    if (state.pB.attackCd <= 0) {
      state.bulletsB.push({
        x: state.pB.x, y: state.pB.y,
        vx: Math.cos(state.pB.facingAngle) * 9.5, vy: Math.sin(state.pB.facingAngle) * 9.5,
        damage: 20, life: 50
      });
      state.pB.attackCd = state.pB.attackSpeed;
    }

    // 子彈判定 (A 射 B)
    for (let i = state.bulletsA.length - 1; i >= 0; i--) {
      const b = state.bulletsA[i];
      b.x += b.vx; b.y += b.vy; b.life--;
      if (Math.hypot(b.x - state.pB.x, b.y - state.pB.y) < state.pB.radius + 3) {
        if (state.pB.invulnerable <= 0) {
          state.pB.hp -= b.damage;
          state.pA.score += 150; // 命中加分
          state.pB.score -= 80;
        }
        state.bulletsA.splice(i, 1);
        continue;
      }
      if (b.life <= 0) state.bulletsA.splice(i, 1);
    }

    // 子彈判定 (B 射 A)
    for (let i = state.bulletsB.length - 1; i >= 0; i--) {
      const b = state.bulletsB[i];
      b.x += b.vx; b.y += b.vy; b.life--;
      if (Math.hypot(b.x - state.pA.x, b.y - state.pA.y) < state.pA.radius + 3) {
        if (state.pA.invulnerable <= 0) {
          state.pA.hp -= b.damage;
          state.pB.score += 150;
          state.pA.score -= 80;
        }
        state.bulletsB.splice(i, 1);
        continue;
      }
      if (b.life <= 0) state.bulletsB.splice(i, 1);
    }

    // 雷射切割判定 (A 割 B)
    if (state.echoesA.length > 0) {
      const ec = state.echoesA[0];
      if (pointToSegmentDist(state.pB.x, state.pB.y, state.pA.x, state.pA.y, ec.x, ec.y) < state.pB.radius + 4) {
        state.pB.hp -= state.pA.resonanceDmg;
        state.pA.score += 15; // 雷射持續切割加分
      }
    }

    // 雷射切割判定 (B 割 A)
    if (state.echoesB.length > 0) {
      const ec = state.echoesB[0];
      if (pointToSegmentDist(state.pA.x, state.pA.y, state.pB.x, state.pB.y, ec.x, ec.y) < state.pA.radius + 4) {
        state.pA.hp -= state.pB.resonanceDmg;
        state.pB.score += 15;
      }
    }

    if (state.pA.invulnerable > 0) state.pA.invulnerable--;
    if (state.pB.invulnerable > 0) state.pB.invulnerable--;
  }

  // 擊殺獎勵
  if (state.pA.hp > 0 && state.pB.hp <= 0) state.pA.score += 1000;
  if (state.pB.hp > 0 && state.pA.hp <= 0) state.pB.score += 1000;

  return { scoreA: state.pA.score, scoreB: state.pB.score, hpA: state.pA.hp, hpB: state.pB.hp };
}

// 執行 100 代演化訓練
console.log("🚀 啟動 AFTERIMAGE 遺傳演算法神經網絡訓練器...");
const POPULATION_SIZE = 40;
const GENERATIONS = 80;
const INPUT_SIZE = 8;
const HIDDEN_SIZE = 12;
const OUTPUT_SIZE = 4;

let population = Array.from({ length: POPULATION_SIZE }, () => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));

for (let gen = 1; gen <= GENERATIONS; gen++) {
  const scores = new Array(POPULATION_SIZE).fill(0);

  // 輪流與基準對手及族群成員對戰
  for (let i = 0; i < POPULATION_SIZE; i++) {
    for (let j = 0; j < 3; j++) {
      const oppIndex = (i + 1 + j) % POPULATION_SIZE;
      const res = simulateMatch(population[i], population[oppIndex]);
      scores[i] += res.scoreA;
      scores[oppIndex] += res.scoreB;
    }
  }

  // 排序選拔
  const ranked = population.map((net, idx) => ({ net, score: scores[idx] }))
    .sort((a, b) => b.score - a.score);

  if (gen % 10 === 0 || gen === 1 || gen === GENERATIONS) {
    console.log(`[世代 Gen ${gen}/${GENERATIONS}] 最高適應度: ${Math.round(ranked[0].score)} | 平均適應度: ${Math.round(scores.reduce((a, b) => a + b) / POPULATION_SIZE)}`);
  }

  // 保留前 20% 精英 (Elitism)，突變生成新一代
  const newPop = [];
  const eliteCount = Math.floor(POPULATION_SIZE * 0.25);
  for (let i = 0; i < eliteCount; i++) {
    newPop.push(ranked[i].net);
  }

  while (newPop.length < POPULATION_SIZE) {
    // 錦標賽選擇父代
    const p1 = ranked[Math.floor(Math.random() * eliteCount)].net;
    newPop.push(p1.mutate(0.08, 0.3));
  }
  population = newPop;
}

// 提取並儲存訓練好的冠軍神經網絡權重
const championBrain = population[0].toJSON();
const fs = require('fs');
fs.writeFileSync('/Users/jason/AI Agent/Games/dd-afterimage/trained_model.json', JSON.stringify(championBrain, null, 2));
console.log("✅ 訓練完成！已成功匯出冠軍神經網絡模型權重：trained_model.json");
