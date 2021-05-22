(function() {
  let github = "https://github.com/shmsw25/AmbigQA"

  let references = {
    "nq": "https://www.mitpressjournals.org/doi/pdf/10.1162/tacl_a_00276",
    "ours": "https://arxiv.org/pdf/2004.10645.pdf",
    "ours-bibtex": "https://shmsw25.github.io/assets/bibtex/min2020ambigqa.txt",
    "dpr": "https://arxiv.org/pdf/2004.04906.pdf",
    "orqa": "https://www.aclweb.org/anthology/P19-1612.pdf",
    "hard-em": "https://www.aclweb.org/anthology/D19-1284.pdf",
    "mscoco": "https://arxiv.org/pdf/1504.00325.pdf"
  };

  $( window ).init(function(){
    $( window ).init(function(){
      let offset = 78;
      $('#container').css('margin-top', offset + 10);
      $('#intro-content').append(`
        <h3>About</h3>
        <p>
          Ambiguity is inherent to open-domain question answering; especially when exploring
          new topics, it can be difficult to ask questions that have a single, unambiguous answer.
          We introduce <span class="em">AmbigQA</span>, a new open-domain question answering task
          which involves predicting a set of question-answer pairs,
          where every plausible answer is paired with a disambiguated rewrite of the original question.
        </p>
        <p>
          To study this task,
          we construct <span class="em">AmbigNQ</span>, a dataset covering
          14,042 questions from NQ-open, an existing open-domain QA benchmark.
          We find that over half of the questions in NQ-open are ambiguous.
          The types of ambiguity are diverse
          and sometimes subtle, many of which are only apparent
          after examining evidence provided by a very large text corpus.
          Visit <a href="explorer.html">Data Explorer</a> to see examples!
        </p>
        <p>
          Details can be found in our paper:
        </p>
        <blockquote>
          Sewon Min, Julian Michael, Hannaneh Hajishirzi, Luke Zettlemoyer.
          <a href="` + references["ours"] + `" target="_blank">"AmbigQA: Answering Ambiguous Open-domain Questions"</a>.
          EMNLP 2020.
          [<a href="` + references["ours-bibtex"] + `" target="_blank">BibTeX</a>]
        </blockquote>
      `);

      // load related things
      //let paper = loadCard("Paper", "");
      //let script = loadCard("Evaluation script", "")
      //$('#intro-content').append(paper + script + "<br />");

      // load download cards
      let readmes = `<span class="readme">
          <a href="` + github + `" target="_blank"><i class="fa fa-github"></i> Evaluation script</a> /
          ` + loadGithubLink("Eval script README", "evaluation-script") + " / " +
          loadGithubLink("Data content README", "dataset-contents") + `</span>`;
      let card1 = loadDownloadCard("Download AmbigNQ (light ver.)",
        "[train/dev] question & answers",
        "data/ambignq_light.zip", "1.1M");
      let card2 = loadDownloadCard("Download AmbigNQ (full ver.)",
        `[train/dev] question, answers, original NQ answer,
        visited Wikipedia pages, used search queries & search results`,
        "data/ambignq.zip", "18M");
      let card3 = loadDownloadCard("Download NQ-open",
        "[train/dev/test] question, NQ answer & associated document",
        "data/nqopen.zip", "3.9M");
      $('#intro-content').append("<h3>Data</h3>" + readmes + "<br /><div class='readme' style='margin-top: 10px;'>" + card1 + card2 + card3 + "</div>")
      //$('#intro-content').append("<h3>Optional Resources</h3><div>" + card4 + card5 + card6 + "</div>")
      $('#intro-content').append(`<h3>Additional Resources</h3><div>
        <!--<span class="readme">` + loadGithubLink("README", "additional-resources") + `</span>-->
        <ul>
          <li>
            Wikipedia DB from 01-20-2020 in sqlite db, consistent to <a href="https://github.com/facebookresearch/DrQA/tree/master/scripts/retriever"
            target="_blank">DrQA</a>
            [<a href="data/docs.db.zip"><i class="fa fa-download"></i> plain text (5.0GB)</a>]
            [<a href="data/docs-html.db.zip"><i class="fa fa-download"></i> html preserved (7.7GB)</a>]
          </li>
          <li>
            Wikipedia DB from 01-20-2020 in .tsv.gz, consistent to <a href="https://github.com/facebookresearch/DPR"
            target="_blank">DPR</a>
            [<a href="data/psgs_w100_20200201.tsv.gz"><i class="fa fa-download"></i> .tsv.gz (4.8GB)</a>]
          </li>
          <!--<li>
            Wikidata dump from 01-20-2020
            [<a href="data/latest-all.json.bz2"><i class="fa fa-download"></i>.json.bz2 (49GB)</a>]
          </li>-->
          <li>
            <strong>Update 07/2020:</strong>
            <a href="` + github + `/tree/master/codes" target="_blank">
              <i class="fa fa-github"></i> Baseline codes (DPR and SpanSeqGen)
            </a> are available now, along with <a href="` + github + `/tree/master/codes#need-preprocessed-data--pretrained-models--predictions" target="_blank">model checkpoints</a>.
          </li>
          <!--<li>
            (Coming soon!) Top 100 retrieved passages for 92K NQ-open questions from
            <a href="https://arxiv.org/pdf/2004.04906.pdf" target="_blank">Dense Passage Retrieval</a>.
          </li>-->
        </ul></div>`);

      $('.panel').width($('#intro-content').width()/3-30);
      $('.panel').css("margin-right", "10px");

      // load references
      $('#intro-content').append(`
        <h3>References</h3>
        <ul><li>If you find our task and data useful, ` + loadCitation("ours", "our paper") + `
        </li><li>To refer to the original Natural Questions, ` + loadCitation("nq", "Kwiatkowski et al. (TACL 2019)") + `
        </li><!--<li>If you use provided retrieved passages, ` + loadCitation("dpr", "Karpukhin et al. (2020)") + `
        </li>--><li>We follow ` + loadPaper("orqa", "Lee et al. (ACL 2019)") + ` to filter open-domain questions from the original Natural Questions,
        and follow ` + loadPaper("hard-em", "Min et al. (EMNLP 2019)") + ` for data split.</li>
        <li>
          We adapt <a href="https://github.com/tylin/coco-caption" target="_blank">Microsoft COCO Caption Evaluation</a> from `+ loadPaper("mscoco", "Chen et al. (2015)") + ` for our evaluation script.
        </li>
      `);

      // load references
      $('#intro-content').append(`
        <h3>Contact</h3>
        <p>
          For any questions about the code or data, please contact
          <a href="https://shmsw25.github.io" target="_blank">Sewon Min</a>
          (<a class="icons-sm email-ic" href="mailto:sewon@cs.washington.edu" target="_blank"><i class="fa fa-envelope-o"></i> Email</a>
          <a class="icons-sm twitter-ic" href="https://twitter.com/sewon__min" target="_blank"><i class="fa fa-twitter-square"></i> Twitter</a>)
          or leave <a href="` + github + `/issues"><i class="fa fa-github"></i> issues</a>.
        </p>
      `);
    });
  });

  function loadDownloadCard(title, text, url, mem) {
    return `<div class="panel panel-default panel-inline download-card" style="float: left">
      <div class="panel-heading"><a href="` + url + `">
        <i class="fa fa-download"></i>  ` + title + ` </a>
      </div>
      <div class="panel-body" style="font-size: 90%">
      ` + text +` (` + mem + `)</div>
    </div>`;
  }

  function loadGithubLink(title, legend) {
    return `<a href="` + github + `/#` + legend + `" target="_blank">` + title + `</a>`;
  }

  function loadCitation(citation, keyword) {
    var text = `cite <a target="_blank" href="` + references[citation] + `">` + keyword + `</a>`;
    if (citation==="ours") {
      text += ` [<a target="_blank" href="` + references[citation+"-bibtex"] + `">BibTeX</a>]`
    }
    return text+".";
  }

  function loadPaper(citation, keyword) {
    return `<a target="_blank" href="` + references[citation][1] + `">` + keyword + `</a>`
  }

})();
