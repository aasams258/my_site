// Code modified from: https://dev.opera.com/articles/html5-canvas-painting
// A bit complex, but works the best from all implementations I tried.
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
      context.lineWidth = 3;
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

      canvas.addEventListener('touchstart', canvasHandler, false);
      canvas.addEventListener('touchmove', canvasHandler, false);
      canvas.addEventListener('touchend', canvasHandler, false);
    }
    init();
  }, false); 
}

// This painting tool works like a drawing pencil which tracks the mouse/touch movements.
// There is a probably a better way to refactor the touch/mouses.
function pencil () {
  var tool = this;
  this.started = false;
  var s_x, s_y;
  // This is called when you start holding down the mouse button.
  // This starts the pencil drawing.
  this.mousedown = function (ev) {
      context.beginPath();
      context.moveTo(ev._x, ev._y);
      tool.started = true;
      s_x = ev._x;
      s_y = ev._y;
  };
  // When you begin the touch.
  this.touchstart = function (ev) {
    context.beginPath();
    context.moveTo(ev._x, ev._y);
    tool.started = true;
    s_x = ev._x;
    s_y = ev._y;
  };

  // This function is called every time you move the mouse.
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
  // This function is called every time you touch.
  this.touchmove = function (ev) {
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

  // This is called when you release the touch.
  this.touchend = function (ev) {
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
  if (ev.touches && ev.touches[0]) {
    ev.clientX = ev.touches[0].clientX;
    ev.clientY = ev.touches[0].clientY;
  } 
  // https://stackoverflow.com/questions/17130395/real-mouse-position-in-canvas
  var rect = canvas.getBoundingClientRect();
  ev._x = (ev.clientX - rect.left) / (rect.right - rect.left) * canvas.width ;
  ev._y= (ev.clientY - rect.top) / (rect.bottom - rect.top) * canvas.height ;
  // Call the event handler of the tool.
  var func = pencil[ev.type];
  if (func) {
    func(ev);
  }
}

function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  var table = document.getElementById("topK");
  for (var i  = 1; i < table.rows.length;) {
    table.deleteRow(i)
  }
}

// https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch
function submitCanvas() {
  document.getElementById("submitBtn").disabled = true;
  var data = canvas.toDataURL('image/png')
  fetch("/digits", {body: data, method: 'POST'}).then(response => response.json()).then(json => {
    var table = document.getElementById("topK");

    // Incase the submit button is pressed multiple times.
    for (var i  = 1; i < table.rows.length;) {
      table.deleteRow(i)
    }

    if (json.error) {
      document.getElementById("error").innerHTML = json.error
    } else {
      for (var i = 0; i < json.prediction.length; i++) {
        var row = table.insertRow(-1);
        var cell1 = row.insertCell(0);
        var cell2 = row.insertCell(1);
        cell1.innerHTML = json.prediction[i][0];
        cell2.innerHTML = Number(json.prediction[i][1]).toFixed(2);
      }
      document.getElementById("transformed").style.visibility = "visible";
      document.getElementById("transformed").src = json.img_uri;
   }
   document.getElementById("submitBtn").disabled = false;    
  })
}