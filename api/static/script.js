const sell_threshold = 0.25;
const j_threshold = 0.4;
const j_boundary = 0.05;
const buy_threshold = 0.65;
const apiUrl = window.location.href;

// change this Value to set the percentage
let totalRot = ((80 / 100) * 180 * Math.PI) / 180;
let animationID = null;
let rotation = 0;
let doAnim = true;
let canvas = null;
let ctx = null;
let text = document.querySelector(".text");
let actionText = document.querySelector(".action");
let errorText = document.querySelector(".error_text");
let tickerInput = document.getElementById("ticker_input");
let waveLoader = document.getElementById("wave_loader");
waveLoader.style.display = "none";
canvas = document.getElementById("canvas");
ctx = canvas.getContext("2d");

document.getElementById("ticker_button").addEventListener(
  "click",
  function () {
    // reset animation
    if (animationID != null) {
      cancelAnimationFrame(animationID);
    }
    rotation = 0;
    errorText.innerText = "";

    // make async request to get probability, once received, start animation
    let ticker = tickerInput.value;

    if (!ticker) {
      errorText.innerText = "Please enter a ticker.";
      return;
    }

    waveLoader.style.display = "flex";

    fetch(apiUrl + "probability/" + ticker)
      .then((response) => {
        waveLoader.style.display = "none";
        if (!response.ok) {
          return Promise.reject(response);
        }
        return response.json();
      })
      .then((data) => {
        totalRot = (data.probability * 180 * Math.PI) / 180;
        animationID = requestAnimationFrame(animate);
        setTimeout(animationID, 1500);
      })
      .catch((response) => {
        try {
          response.json().then((json) => {
            errorText.innerText = json.detail;
          });
        } catch (err) {
          errorText.innerText = `Failed to load data for '${ticker}'.`;
        }
      });
  },
  false
);

function calcPointsCirc(cx, cy, rad, dashLength) {
  var n = rad / dashLength,
    alpha = (Math.PI * 2) / n,
    pointObj = {},
    points = [],
    i = -1;

  while (i < n) {
    var theta = alpha * i,
      theta2 = alpha * (i + 1);

    points.push({
      x: Math.cos(theta) * rad + cx,
      y: Math.sin(theta) * rad + cy,
      ex: Math.cos(theta2) * rad + cx,
      ey: Math.sin(theta2) * rad + cy,
    });
    i += 2;
  }
  return points;
}

function animate() {
  //Clearing animation on every iteration
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const center = {
    x: 175,
    y: 175,
  };

  // main arc
  ctx.beginPath();
  if (rotation < sell_threshold * Math.PI) {
    // strong sell
    ctx.strokeStyle = "#FF0000";
  } else if (rotation < (j_threshold - j_boundary) * Math.PI) {
    // sell
    ctx.strokeStyle = "#FFCCCB";
  } else if (rotation < (j_threshold + j_boundary) * Math.PI) {
    // neutral
    ctx.strokeStyle = "#C0C0C0";
  } else if (rotation < buy_threshold * Math.PI) {
    // buy
    ctx.strokeStyle = "#90EE90";
  } else {
    // strong buy
    ctx.strokeStyle = "#00FF00";
  }
  ctx.lineWidth = "3";
  let radius = 174;
  ctx.arc(center.x, center.y, radius, Math.PI, Math.PI + rotation);
  ctx.stroke();

  // Green Arc
  if (rotation <= buy_threshold * Math.PI) {
    ctx.beginPath();
    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = "3";
    ctx.arc(center.x, center.y, radius, (1 + buy_threshold) * Math.PI, 0);
    ctx.stroke();
  }

  //functions to draw dotted lines
  const DrawDottedLine = (x1, y1, x2, y2, dotRadius, dotCount, dotColor) => {
    var dx = x2 - x1;
    var dy = y2 - y1;
    let slopeOfLine = dy / dx;
    var degOfLine =
      Math.atan(slopeOfLine) * (180 / Math.PI) > 0
        ? Math.atan(slopeOfLine) * (180 / Math.PI)
        : 180 + Math.atan(slopeOfLine) * (180 / Math.PI);
    var degOfNeedle = rotation * (180 / Math.PI);

    let defaultColor = "#708090";
    if (rotation < sell_threshold * Math.PI) {
      // strong sell
      dotColor = degOfLine <= degOfNeedle ? "#FF0000" : defaultColor;
    } else if (rotation < (j_threshold - j_boundary) * Math.PI) {
      // sell
      dotColor = degOfLine <= degOfNeedle ? "#FFCCCB" : defaultColor;
    } else if (rotation < (j_threshold + j_boundary) * Math.PI) {
      // neutral
      dotColor = degOfLine <= degOfNeedle ? "#C0C0C0" : defaultColor;
    } else if (rotation < buy_threshold * Math.PI) {
      // buy
      dotColor = degOfLine <= degOfNeedle ? "#90EE90" : defaultColor;
    } else {
      // strong buy
      dotColor = degOfLine <= degOfNeedle ? "#00FF00" : defaultColor;
    }

    var spaceX = dx / (dotCount - 1);
    var spaceY = dy / (dotCount - 1);
    var newX = x1;
    var newY = y1;
    for (var i = 0; i < dotCount; i++) {
      dotRadius = dotRadius >= 0.75 ? dotRadius - i * (0.5 / 15) : dotRadius;
      drawDot(newX, newY, dotRadius, `${dotColor}${100 - (i + 1)}`);
      newX += spaceX;
      newY += spaceY;
    }
  };
  const drawDot = (x, y, dotRadius, dotColor) => {
    ctx.beginPath();
    ctx.arc(x, y, dotRadius, 0, 2 * Math.PI, false);
    ctx.fillStyle = dotColor;
    ctx.fill();
  };
  let firstDottedLineDots = calcPointsCirc(center.x, center.y, 165, 1);
  for (let k = 0; k < firstDottedLineDots.length; k++) {
    let x = firstDottedLineDots[k].x;
    let y = firstDottedLineDots[k].y;
    DrawDottedLine(x, y, 175, 175, 1.75, 30, "#35FFFF");
  }

  //dummy circle to hide the line connecting to center
  ctx.beginPath();
  ctx.arc(center.x, center.y, 80, 2 * Math.PI, 0);
  ctx.fillStyle = "black";
  ctx.fill();

  //Speedometer triangle
  var x = -75,
    y = 0;
  ctx.save();
  ctx.beginPath();
  ctx.translate(175, 175);
  ctx.rotate(rotation);
  ctx.moveTo(x, y);
  ctx.lineTo(x + 10, y - 10);
  ctx.lineTo(x + 10, y + 10);
  ctx.closePath();
  ctx.fillStyle = "#FF9421";
  ctx.fill();
  ctx.restore();
  let stopAnimation = false;
  if (rotation < totalRot) {
    rotation += (1 * Math.PI) / 180;
    if (rotation > totalRot) {
      rotation -= (1 * Math.PI) / 180;
      stopAnimation = true;
    }
  }

  if (rotation < sell_threshold * Math.PI) {
    actionText.innerHTML = "Strong Sell";
    actionText.style.color = "#FF0000";
  } else if (rotation < (j_threshold - j_boundary) * Math.PI) {
    actionText.innerHTML = "Sell";
    actionText.style.color = "#FFCCCB";
  } else if (rotation < (j_threshold + j_boundary) * Math.PI) {
    actionText.innerHTML = "Neutral";
    actionText.style.color = "#C0C0C0";
  } else if (rotation < buy_threshold * Math.PI) {
    actionText.innerHTML = "Buy";
    actionText.style.color = "#90EE90";
  } else {
    actionText.innerHTML = "Strong Buy";
    actionText.style.color = "#00FF00";
  }

  text.innerHTML = Math.round((rotation / Math.PI) * 100) + 0 + "%";
  if (!stopAnimation) {
    animationID = requestAnimationFrame(animate);
  }
}
