(function() {

  var allData;

  let ENCODETITLE = [" ", "%", "!", "#", "$", "&", "'", "(", ")", "*", "+", ",", "/", ":", ";", "=",
    "?", "@", "[", "]"];
  let ENCODEURL = ["_", "%25", "%21", "%23", "%24", "%26", "%27", "%28", "%29",
    "%2A"< "%2B", "%2C", "%2F", "%3A", "%3B", "%3D", "%3F", "%40", "%5B", "%5D"];

  $( window ).init(function(){
    $("body").css("overflow", "hidden");
    $("#container").css("max-width", "2400px");
    $('.row').width($('#container').width());
    let offset = 78 + $('#second-navbar').height();
    $('.row').css('margin-top', offset + 10);
    $('.col').css('height', $( window ).height() - offset - 10);
    $('#content').html("Loading data...");
    loadAllData();
  });

  function loadAllData(){
    $('.textbox').width($('#sidebar').width());
    $.getJSON("data/dev_small.json", function(json) {
      allData = json;
      loadData();
    });
    /*
    sendAjax("/select", {}, (result) => {
      origData = result;
      allData = result.data;
      loadData();
    });*/
  }

  function loadData() {
    $('#sidebar').html('');
    $('#content').html('');
    for (var i=0; i<allData.length; i++) {
      let htmlText = `
      <div id="textbox-` + i + `" class="textbox ` + getClassName(allData[i]) + `">
      ` + allData[i]['question'] +
      `</div>
      `;
      $('#sidebar').append(htmlText);
    }
    $('.textbox').click(function () {
      $('.textbox-clicked').removeClass('textbox-clicked');
      $('#'+this.id).addClass('textbox-clicked');
      load(parseInt(this.id.substring(8, this.id.length)));
    })
    $('.mode').click(controlDisplay);
    controlDisplay();
  }

  function controlDisplay() {
    let mode = parseInt($('.mode:checked').val());
    ('.textbox-clicked');
    if (mode===0) {
      $('.multiple-dp').show();
      $('.single-dp').hide();
      if (getCurrentClassName()==="single-dp") {
        $('.textbox-clicked').removeClass('textbox-clicked');
        $('#content').html('');
      }
    } else if (mode===1) {
      $('.multiple-dp').hide();
      $('.single-dp').show();
      if (getCurrentClassName()==="multiple-dp") {
        $('.textbox-clicked').removeClass('textbox-clicked');
        $('#content').html('');
      }
    } else {
      $('.multiple-dp').show();
      $('.single-dp').show();
    }
   }

  function getClassName(data) {
    //FIXME later
    if (data['annotations'].map(ann => ann['type']==='singleAnswer').every(x => x)) {
      return "single-dp"
    } else if (data['annotations'].map(ann => ann['type']==='multipleQAs').every(x => x)) {
      return "multiple-dp"
    }
  }

  function getCurrentClassName() {
    let current = $('.textbox-clicked');
    console.assert(current.length<=1);
    if (current.length===1) {
      let currentId = current[0].id;
      return getClassName(allData[parseInt(currentId.substring(8, currentId.length))]);
    }
    return "";
  }

  function load(currentId) {
    $('#content').html("");
    let data = allData[currentId];
    var annotations = data['annotations'];
    $('#content').append(getPanel("Prompt Question", data["question"]));

    for (var i=0; i<annotations.length; i++) {
      var htmlText = "";
      if (annotations[i]['type']==='multipleQAs') {
        var qaPairs = annotations[i]['qaPairs'];
        for (var j=0; j<qaPairs.length; j++) {
          let pair = qaPairs[j];
          htmlText += `
              <p><span class="label label-primary">Question</span> ` + pair['question'] + `</p>
              <p><span class="label label-info">Answer</span> ` + pair['answer'].join(" | ") + `</p>
          `;
        }
      } else if (annotations[i]['type']==='singleAnswer') {
          htmlText = `
              <p><span class="label label-info">Answer</span> ` + annotations[i]['answer'].join(" | ") + `</p>
          `;
      } else {
        htmlText = `<em>Answer not found</em>`;
      }
      $('#content').append(getPanel("Annotation #" + (i+1).toString(), htmlText));
    }
    var titleText = "";
    for (var i=0; i<data['viewed_doc_titles'].length; i++) {
      let title = data['viewed_doc_titles'][i];
      titleText += `
        <span class='label label-simple'>
          <a href='` + getWikiURL(title) + `' target='_blank'>` + title + `</a>
        </span>`;
    }
    $('#content').append(getPanel("Wikipedia pages visited by annotators", titleText));
    $('#content').append(getPanel("Original NQ answer", data["nq_answer"].join(" | ")));
  }

  function getPanel(heading, body) {
    return `<div class="panel panel-default panel-inline">
      <div class="panel-heading">` + heading + `
      </div>
      <div class="panel-body">
      ` + body +`</div>
    </div>`;
  }

  function getWikiURL(title) {
    var encoded_title = title;
    for (var i=0; i<ENCODETITLE.length; i++) {
      encoded_title = encoded_title.replace(ENCODETITLE[i], ENCODEURL[i]);
    }
    return "https://en.wikipedia.org/wiki/" + encoded_title;
  }

  function sendAjax(url, data, handle){
    $.getJSON(url, data, function(response){
      handle(response.result);
    });
  }



})();
