const target = [
  [5.1, 3.5, 1.4, 0.2],
  [0, 3.1, 1.5, 0.1],
  [5.1, 0, 1.5, 0.3],
  [5.2, 4.1, 1.5, 0],
  [0, 2.7, 3.9, 1.4],
  [5.6, 0, 4.5, 0],
  [5.5, 2.6, 0, 1.2],
  [5.1, 2.5, 3.0, 0],
  [5.7, 0, 5.0, 2.0],
  [0, 2.8, 6.1, 1.9],
  [5.8, 2.7, 5.1, 0],
  [6.5, 0, 0, 2.0],
];

const std = [0.648, 0.193, 3.116, 0.559];
const mean = [5.865, 3.04, 3.718, 1.191];

let result = [];

// for (i = 0; i <= 11; i++) {
//   for (j = 0; j < 4; j++) {
//     if (flag[i][j] !== 0) {
//       flag[i][j] -= mean[j];
//     }
//   }
// }

for (i = 0; i <= 11; i++) {
  for (j = 0; j < 4; j++) {
    if (target[i][j] !== 0) {
      result.push(Math.abs(mean[j] - target[i][j]) / std[j]);
    }
  }
}

console.log(result.length);

let alpha =
  result.reduce((pre, cur) => {
    return pre + cur;
  }) / result.length;

console.log(alpha);
