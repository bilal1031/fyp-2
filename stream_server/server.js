var app = require("express")();
var http = require("http").createServer(app);
var io = require("socket.io")(http);
var sizeof = require("object-sizeof");

try {
  let frames = [];

  app.get("/", function (req, res) {
    res.send("running");
  });
  const videoReciever = io
    .of("videoReciever")
    .on("connection", function (socket) {
      console.log(`${socket.nsp?.name}: new connection with id '${socket.id}'`);
    });

  const videoSender = io.of("videoSender").on("connection", function (socket) {
    console.log(`${socket.nsp?.name}: new connection with id '${socket.id}'`);

    socket.on("data", function (data) {
      // listen on client emit 'data'
      var ret = Object.assign({}, data, {
        frame: Buffer.from(data, "base64").toString(), // from buffer to base64 string
      });
      console.clear();
      console.log(ret?.frame.length);
      if (ret.frame.length > 0) {
        frames.push(ret.frame);
        videoReciever.emit("data", `data:image/jpeg;base64,${frames.shift()}`);
      }
    });
  });

  videoSender.on("disconnect", function (socket) {
    console.log(`${socket.nsp?.name}: disconnected with id '${socket.id}'`);
  });

  http.listen(3000, "localhost", function () {
    console.log("listening on :3000");
  });
} catch (error) {
  console.log(error);
}
