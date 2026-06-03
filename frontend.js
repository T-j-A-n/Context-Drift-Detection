const ws = new WebSocket("ws://localhost:8000/ws");

ws.onopen = () => {
  ws.send(JSON.stringify({
    context: "Q2 sales strategy",
    tolerance: 0.4
  }));
};

// send audio chunks
mediaRecorder.ondataavailable = (event) => {
  ws.send(event.data);
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  console.log("Drift:", data.drift);

  if (data.alert) {
    console.log("🚨 Drift detected!");
  }
};