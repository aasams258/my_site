/* © 2009 ROBO Design
 * http://www.robodesign.ro
 */
// https://dev.opera.com/articles/html5-canvas-painting/

// Keep everything in anonymous function, called on window load.

// Refactor this so its not so wrapped up, and only reads mouse events on the canvas itself.
var canvas, context, pencil;
if(window.addEventListener) {
  window.addEventListener('load', function () {
    function init () {
      // Find the canvas element.
      canvas = document.getElementById('imageView');
      if (!canvas) {
        alert('Error: I cannot find the canvas element!');
        return;
      }

      if (!canvas.getContext) {
        alert('Error: no canvas.getContext!');
        return;
      }

      // Get the 2D canvas context.
      context = canvas.getContext('2d');
      context.lineWidth= 3;
      if (!context) {
        alert('Error: failed to getContext!');
        return;
      }

      // Pencil tool instance.
      pencil = new pencil();

      // Attach the mousedown, mousemove and mouseup event listeners.
      canvas.addEventListener('mousedown', canvasHandler, false);
      canvas.addEventListener('mousemove', canvasHandler, false);
      canvas.addEventListener('mouseup', canvasHandler, false);
      canvas.addEventListener('mouseleave', canvasHandler, false);
    }
    init();
  }, false); 
}
// This painting tool works like a drawing pencil which tracks the mouse 
// movements.
function pencil () {
  var tool = this;
  this.started = false;
  var s_x, s_y;
  // This is called when you start holding down the mouse button.
  // This starts the pencil drawing.
  this.mousedown = function (ev) {
    // TODO make it so a press down draws a dot.
      context.beginPath();
      context.moveTo(ev._x, ev._y);
      tool.started = true;
      s_x = ev._x;
      s_y = ev._y;
  };

  // This function is called every time you move the mouse. Obviously, it only 
  // draws if the tool.started state is set to true (when you are holding down 
  // the mouse button).
  this.mousemove = function (ev) {
    if (tool.started) {
      if (ev._x == s_x && ev._y == s_y) {
        context.lineTo(ev._x + 1, ev._y + 1);
        context.stroke();
      } else {
        context.lineTo(ev._x, ev._y);
        context.stroke();
      }
    }
  };

  // This is called when you release the mouse button.
  this.mouseup = function (ev) {
    if (tool.started) {
      tool.mousemove(ev);
      tool.started = false;
    }
  };
  this.mouseleave = function (ev) {
    if (tool.started) {
      tool.started = false;
    }
  };
}
    // The general-purpose event handler. This function just determines the mouse 
    // position relative to the canvas element.
function canvasHandler (ev) {
  if (ev.layerX || ev.layerX == 0) { // Firefox
    ev._x = ev.layerX;
    ev._y = ev.layerY;
  } else if (ev.offsetX || ev.offsetX == 0) { // Opera
    ev._x = ev.offsetX;
    ev._y = ev.offsetY
  }

  // Call the event handler of the tool.
  var func = pencil[ev.type];
  if (func) {
    func(ev);
  }
}

function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById("response").innerHTML = ""
}

// https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest/send
// https://stackoverflow.com/questions/6396101/pure-javascript-send-post-data-without-a-form
// https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
function submitCanvas() {
  var data = canvas.toDataURL('image/png')
  fetch("/draw", {body: data, method: 'POST'}).then(response => response.json()).then(json => {
    console.log(json.prediction)
    document.getElementById("response").innerHTML = "Prediction: " + json.prediction
  })
  console.log("Submitted")
  
}