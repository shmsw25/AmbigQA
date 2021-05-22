(function() {
  let ourPaperURL = "https://arxiv.org/abs/2004.10645";
  let ourModelName = "SpanSeqGen";
  let github = "https://github.com/shmsw25/AmbigQA";

  var data = [
    {
      "date": "Oct 7, 2020",
      "model": "Refuel (ensemble)",
      "institution": "AWS AI",
      "paper": "Gao et al. ACL 2021",
      "paper-url": "https://arxiv.org/abs/2011.13137",
      "result": ["<strong>44.3</strong>", "<strong>34.8</strong>",
        "<strong>15.9</strong>", "<strong>10.1</strong>"]
    },
    {
      "date": "Sep 17, 2020",
      "model": "Refuel (single model)",
      "institution": "AWS AI",
      "paper": "Gao et al. ACL 2021",
      "paper-url": "https://arxiv.org/abs/2011.13137",
      "result": ["42.1", "33.3", "15.3", "9.6"]
    },
    {
      "date": "Apr 20, 2020",
      "model": ourModelName + " (Co-training)",
      "institution": "University of Washington",
      "paper": "Min et al. EMNLP 2020",
      "paper-url": ourPaperURL,
      "result": ["35.9", "26.0", "11.5", "6.3"]
    },
    {
      "date": "Apr 20, 2020",
      "model": ourModelName + " (Ensemble)",
      "institution": "University of Washington",
      "paper": "Min et al. EMNLP 2020",
      "paper-url": ourPaperURL,
      "result": [35.2, 24.5, 10.6, 5.7]
    },
    {
      "date": "Apr 20, 2020",
      "model": ourModelName,
      "institution": "University of Washington",
      "paper": "Min et al. EMNLP 2020",
      "paper-url": ourPaperURL,
      "result": [33.5, 24.5, 11.4, 5.8]
    }
  ];
  var zeroshotData = [
    {
      "date": "Apr 20, 2020",
      "model": ourModelName,
      "institution": "University of Washington",
      "paper": "Min et al. EMNLP 2020",
      "paper-url": ourPaperURL,
      "result": ["<strong>42.2</strong>", "<strong>30.8</strong>", 20.7]
    },
    {
      "date": "Apr 20, 2020",
      "model": "Dense Passage Retrieval (DPR)",
      "institution": "Facebook AI Research",
      "paper": "Karpukhin et al. EMNLP 2020",
      "paper-url": "https://arxiv.org/abs/2004.04906",
      "result": [41.5, 30.1, "<strong>23.2</strong>"]
    },
    {
      "date": "Apr 20, 2020",
      "model": "PathRetriever",
      "institution": "Salesforce Research + University of Washington",
      "paper": "Asai et al. ICLR 2020",
      "paper-url": "https://arxiv.org/abs/1911.10470",
      "result": [32.6, 27.9, 17.7]
    },
    {
      "date": "Apr 20, 2020",
      "model": "GraphRetriever",
      "institution": "University of Washington",
      "paper": "Min et al. 2019",
      "paper-url": "https://arxiv.org/abs/1911.03868",
      "result": [34.5, 27.5, "17.0"]
    }
  ]

  $( window ).init(function(){
    $( window ).init(function(){
      //$('#leaderboard-table').width($('#container').width());
      $("#container").css("max-width", "2400px");
      $('.row').width($('#container').width());
      let offset = 78;
      $('.row').css('margin-top', offset + 10);
      $('.col').css('height', $( window ).height() - offset - 10);
      $('#content').append(`
        <div style="text-align: center;"><h3>Standard setting</h3></div>
        <table id="leaderboard-table" class="table"></table>
        <div style="text-align: center;"><h3>Zero-shot setting</h3></div>
        <table id="zeroshot-leaderboard-table" class="table"></table>
      `);
      loadTable();
      $('#sidebar').css('padding', '20px');
      $('#sidebar').css('padding-top', '30px');
      $("#sidebar").append(`
        <h3>Settings</h3>
            We have two settings, <em>Standard</em> and <em>Zero-shot</em>
            which <em>can</em> and <em>cannot</em> access the train set of AmbigNQ, respectively.
        <hr />
        <h3>Evaluation</h3>
            <em>F1 answer</em> considers multiple answer prediction only, and
            <em>F1 bleu</em> & <em>F1 edit-f1</em> consider the full task.
            Please see the <a href="` + ourPaperURL + `" target="_blank">paper</a> for the full definition.
        <hr />
        <h3>Leaderboard Submission</h3>
            To submit your model, please see
            <a href="` + github + `/#leaderboard-submission-guide" target="_blank">submission guide</a>.
        `)
    })
  });

  function loadTable(){
    $('#leaderboard-table').append(`
      <thead>
        <tr>
          <th width="5%">Rank</th>
          <th width="55%">Model</th>
          <th width="10%">F1 answer (<em>all</em>)</th>
          <th width="10%">F1 answer (<em>multi</em>)</th>
          <th width="10%">F1 bleu</th>
          <th width="10%">F1 edit-f1</th>
        </tr>
      <thead>
    `);
    var tbody = ""
    for (var i=0; i<data.length; i++) {
      tbody += loadRow(i, data);
    }
    $('#leaderboard-table').append("<tbody>" + tbody + "</tbody>");
    $('#zeroshot-leaderboard-table').append(`
      <thead>
        <tr>
          <th width="5%">Rank</th>
          <th width="65%">Model</th>
          <th width="10%">NQ-open EM</th>
          <th width="10%">F1 answer (<em>all</em>)</th>
          <th width="10%">F1 answer (<em>multi</em>)</th>
        </tr>
      <thead>
    `);
    var tbody = ""
    for (var i=0; i<zeroshotData.length; i++) {
      tbody += loadRow(i, zeroshotData);
    }
    $('#zeroshot-leaderboard-table').append("<tbody>" + tbody + "</tbody>");
  }

  function loadRow(i, data) {
    let className = (i%2===0)? "gray" : "";
    var tbody = `<tr class="` + className + `">
      <td>` + (i+1) + `<br /><span class="label lable-default date">` + data[i]["date"] + `</span></td>
      <td><span class="modelname">` + data[i]["model"] + `</span><br />
      <span class="institution">` + data[i]["institution"] + `</span><br />`;
    if (data[i]["paper"].length>0 && data[i]["paper-url"].length>0) {
      tbody += `<a class="paper" target="_blank" href="` + data[i]["paper-url"] + `">` + data[i]["paper"] + `</a>`;
    }
    tbody += "</td>";
    for (var j=0; j<data[i]["result"].length; j++) {
      tbody += `<td>` + data[i]["result"][j] + `</td>`;
    }
    tbody += "</tr>";
    return tbody;
  }

})();
