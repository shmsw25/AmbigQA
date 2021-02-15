(function() {

  var question_id;
  var prompt_question;
  var PAY_BASE = 10;
  var PAY_DELTA = 10;
  var PAY_MULTIQA = 20;
  var currentPay = 0;
  var cache = [];
  var imgURLPrefix = "http://shmsw25.github.io/intention-annotation"; //"https://shmsw25.github.io/assets/screenshots/"

  /* main code start */

  var instructions;
  let task = $('#taskKey').attr("value");
  console.log(task);
  $( window ).init(function(){
    if (task=='"qualification"') {
      instructions = updateQualificationInstructions(getValidationInstructions());
      loadQualificationHTML();
      setWidth();
      loadInstructions(instructions);
      setUpForQualification();
    } else if (task=='"validation"') {
      instructions = getValidationInstructions();
      loadValidationHTML();
      setWidth();
      loadInstructions(instructions);
      setPromptAnnotations();
      setUpForValidation();
    } else if (task==='"final"') {
      loadFinalHTML();
      setWidth();
      loadAll(null);
      let valPrompt = JSON.parse($('#prompt').attr("value"))["validationPrompt"];
      if (valPrompt["generationPrompt"]["question"]=="none") {
        console.log("Error");
        $('#container').html("<span class='red large bold'>Technical problem: Please skip/return your HIT.</span>")
        return;
      }
      var promptAnnotations = JSON.parse($('#prompt').attr("value"))['annotations'];
      for (i in promptAnnotations) {
        let promptAnnotation = promptAnnotations[i];
        var htmlText = "";
        var titles = [];
        if (promptAnnotation['type']==='noAnswer') {
          htmlText += "No answer found";
        } else if (promptAnnotation['type']==='singleAnswer') {
          htmlText += "Single clear answer<p><span class='graybackground'>A:</span> ";
          htmlText += promptAnnotation['answer'] + "</p>";
          titles = titles.concat(promptAnnotation['titles'])
          console.log(promptAnnotation, titles);
        } else if (promptAnnotation['type']==='multipleQAs') {
          htmlText += "Multiple question-answer pairs</p>";
          promptAnnotation['qaPairs'].forEach( d => {
            htmlText += "<p><span class='graybackground'>Q:</span> " + d['question'] + "<br />";
            htmlText += "<span class='graybackground'>A:</span> " + d['answer'] + "</p>";
            titles = titles.concat(d['titles']);
            //console.log(d['titles']);
          } );
        }
        $('#promptAnnotationsDiv').append(`
          <div class="panel panel-default panel-inline">
            <div class="panel-heading">Option #` + (parseInt(i)+1).toString() + `
              <input type="checkbox" class="custom-control-input correct-checkbox" id="correct-checkbox-` + i + `"></input> Correct?<br />
            </div>
            <br />
            <div class="panel-body" id="prompt-annotations-` + i + `">` + htmlText + `
            </div>
          </div>
          `);
        console.log(titles);
        appendTitles(titles, "#prompt-annotations-"+i);
      }
      $('.hyperlink').css('font-size', '11pt');
      var responses = [];
      for (var i=0; i<promptAnnotations.length; i++) {
        responses.push(0);
      }
      $('#response').val(JSON.stringify(responses));
      $('.correct-checkbox').change(function () {
        var responses = []
        for (var i=0; i<promptAnnotations.length; i++) {
          responses.push(($('#correct-checkbox-'+i).prop('checked'))? 1 : 0);
        }
        $('#response').val(JSON.stringify(responses));
      });
      /*
      $("#edit-mode").click(function () {
        var responses = []
        for (var i=0; i<promptAnnotations.length; i++) {
          responses.push(($('#correct-checkbox-'+i).prop('checked'))? 1 : 0);
        }
        if (responses.reduce((a,b)=>a+b, 0) !== 1) {
          alert("Please check *one* of annotations for the default annotation.");
        } else {

        }
      });*/
    } else if (task.startsWith('"generation')) {
      instructions = getGenerationInstructions();
      loadGenerationHTML();
      setWidth();
      loadAll(instructions);
      loadInitAnnotationForm();
    }
    //turkSetAssignmentID();
    $('#container').append(`
      <button type="submit" disabled id="actual-submit" style="display:none""></button>`);
    $('#submit').click(function(){
      $('#actual-submit').prop('disabled', false);
      //$('#actual-submit').show();
      document.getElementById("actual-submit").click();
      console.log("clicked");
    })
  });

  function sendAjax(url, data, handle){
		$.getJSON(url, data, function(response){
			handle(response.result);
		});
	}


  function loadGenerationHTML() {
    $('#taskContent').html(
    `<div class="container" id="container" role="main"><div class="panel panel-default">
        <div class="panel-heading"><button id="instruction-header" type="button" class="btn-lg" ></button></div>
        <div class="panel-body" id="instruction-body">
          <nav class="navbar navbar-default">
            <div class="container-fluid">
              <ul class="nav navbar-nav">
                <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                <li><a href="#" id="instructions-step-by-step" class="instructions-item">Examples</a></li>
                <li><a href="#" id="instructions-examples" class="instructions-item">FAQ</a></li>
                <li><a href="#" id="instructions-bonuses" class="instructions-item">Bonuses</a></li>
              </ul>
            </div>
          </nav>
          <div id="instructions">
            Instructions (TODO)
          </div>
        </div>
      </div>
      <div class="row">
      <div class="col col-12 col-md-8">
        <div class="panel panel-default narrow-panel">
          <div class="panel-heading">Input Question</div>
          <div class="panel-body" id="input-question">(Loading...)</div>
        </div>
        <!-- Input box for User Input -->
        <div class="input-group narrow-input-group" id="write-question">
          <input class="editOption editable" id="search-query" placeholder="Write Query for Search Wikipedia" />
          <span class="input-group-addon btn btn-default run" id="search-button">Search</span>
          <br />
        </div>
        <div id="search-results"></div>
        <button type="button" class="btn btn-primary go-back" id="go-back-to-search-results" style="display:none;">< Go back to search results</button>
        <div id="wikipedia-box" class="narrow-wikipedia-box"></div>
      </div>
      <div class="col col-8 col-md-4">
        <div id="checkboxes">
          <p>
            <input type="checkbox" class="custom-control-input" id="single-clear-answer-checkbox"></input>  Single clear answer?<br />
          </p>
          <p>
            <input type="checkbox" class="custom-control-input" id="answer-not-found-checkbox"></input>  Answer not found?<br />
          </p>
        </div>
        <div id="annotations">
        </div>
        <div id="buttons">
          <button type="button" class="btn btn-default" id="add-pair-button">Add pair</button>
          <button type="button" class="btn btn-primary" id="submit">Submit!</button>
          <p id="pay-hint" class="small-hint"></p>
          <p id="validated-hint" class="small-hint red"></p>
          <div id="warning-box-input-type-1" class="warning-box" style="display:none"></div>
          <div id="warning-box-input-type-2" class="warning-box" style="display:none"></div>
          <div id="warning-box-input-type-3" class="warning-box" style="display:none"></div>
          <div id="warning-box-input-type-4" class="warning-box" style="display:none"></div>
          <div id="warning-box-input-type-5" class="warning-box" style="display:none"></div>
          <div id="guide-box-response-type" class="guide-box"></div>
          <div id="guide-box-input-type" class="guide-box" style="display:none"></div>
        </div>
        <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
        <br /><br />
        <input type="checkbox" class="custom-control-input" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
      </div>
    </div>
  </div>`);
  }


  function loadValidationHTML() {
    /*$('#taskContent').html(
      `<div class="container" id="container" role="main"><div class="panel panel-default">
        <div class="panel-heading"><button id="instruction-header" type="button" class="btn-lg" ></button></div>
        <div class="panel-body" id="instruction-body">
          <nav class="navbar navbar-default">
            <div class="container-fluid">
              <ul class="nav navbar-nav">
                <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                <li><a href="#" id="instructions-step-by-step" class="instructions-item">Step-by-step instructions</a></li>
                <li><a href="#" id="instructions-examples" class="instructions-item">FAQ</a></li>
                <li><a href="#" id="instructions-bonuses" class="instructions-item">Bonuses</a></li>
              </ul>
            </div>
          </nav>
          <div id="instructions">
            Instructions (TODO)
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col col-12 col-md-8">
          <div class="panel panel-default narrow-panel">
            <div class="panel-heading">Input Question</div>
            <div class="panel-body" id="input-question">(Loading...)</div>
          </div>
          <!-- Input box for User Input -->
          <div class="input-group narrow-input-group" id="write-question">
            <input class="editOption editable" id="search-query" placeholder="Write Query for Search Wikipedia" />
            <span class="input-group-addon btn btn-default run" id="search-button">Search</span>
            <br />
          </div>
          <div id="search-results"></div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-search-results" style="display:none;">< Go back to search results</button>
          <div id="wikipedia-box" class="narrow-wikipedia-box"></div>
        </div>
        <div class="col col-8 col-md-4">
          <div class="panel panel-default panel-inline">
            <div class="panel-heading"><button id="prompt-annotations-header" type="button" class="btn-lg"></button></div>
            <div class="panel-body" id="prompt-annotations">
            </div>
          </div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-default-annotation">Go back to default input</button>
          <div id="checkboxes">
            <input type="checkbox" class="custom-control-input" id="single-clear-answer-checkbox"></input>  Single clear answer? <br />
            <input type="checkbox" class="custom-control-input" id="answer-not-found-checkbox"></input>  Answer not found? <br />
          </div>
          <div id="annotations">
          </div>
          <div id="buttons">
            <button type="button" class="btn btn-default" id="add-pair-button">Add pair</button>
            <button type="submit" class="btn btn-primary" disabled id="submit-button">Submit annotations</button>
            <p id="pay-hint" class="small-hint"></p>
            <div id="warning-box-input-type-1" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-2" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-3" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-4" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-5" class="warning-box" style="display:none"></div>
            <p id="validated-hint" class="small-hint red"></p>
            <div id="guide-box-input-type" class="guide-box" style="display:none"></div>
          </div>
          <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
          <br />
          <br />
          <input type="checkbox" class="custom-control-input" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
        </div>
      </div>
    </div>`);*/
    $('#taskContent').html(
      `<div class="container" id="container" role="main"><div class="panel panel-default">
        <div class="panel-heading"><button id="instruction-header" type="button" class="btn-lg" ></button></div>
        <div class="panel-body" id="instruction-body">
          <nav class="navbar navbar-default">
            <div class="container-fluid">
              <ul class="nav navbar-nav">
                <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                <li><a href="#" id="instructions-step-by-step" class="instructions-item">Step-by-step instructions</a></li>
                <li><a href="#" id="instructions-examples" class="instructions-item">FAQ</a></li>
                <li><a href="#" id="instructions-bonuses" class="instructions-item">Bonuses</a></li>
              </ul>
            </div>
          </nav>
          <div id="instructions">
            Instructions (TODO)
          </div>
        </div>
      </div>
      <div class="row">
        <div class="col">
          <div class="panel panel-default narrow-panel">
            <div class="panel-heading">Input Question</div>
            <div class="panel-body" id="input-question">(Loading...)</div>
          </div>
          <!-- Input box for User Input -->
          <div class="input-group narrow-input-group" id="write-question">
            <input class="editOption editable" id="search-query" placeholder="Write Query for Search Wikipedia" />
            <span class="input-group-addon btn btn-default run" id="search-button">Search</span>
            <br />
          </div>
          <div id="search-results"></div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-search-results" style="display:none;">< Go back to search results</button>
          <div id="wikipedia-box" class="narrow-wikipedia-box"></div>
        </div>
        <div class="col side">
          <div class="panel panel-default panel-inline">
            <div class="panel-heading"><button id="prompt-annotations-header" type="button" class="btn-lg"></button></div>
            <div class="panel-body" id="prompt-annotations">
            </div>
          </div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-default-annotation">Go back to default input</button>
          <button type="button" class="btn btn-primary go-back" id="go-to-step1" style="display:none">Go back to step1</button>
          <button type="button" class="btn btn-primary go-back" id="go-to-step2">Go to step2</button>
          <button type="button" class="btn btn-success" disabled id="submit">Submit annotations</button>
          <p id="description-of-step1">
            Step 1: combine, delete, separate, or correct the answer.
          </p>
          <p id="description-of-step2" style="display:none">
            Step 2: lightly edit or add questions to allow a single clear answer; check questions to exclude.
          </p>
          <!--<div id="checkboxes">
            <input type="checkbox" class="custom-control-input" id="single-clear-answer-checkbox"></input>  Single clear answer? <br />
            <input type="checkbox" class="custom-control-input" id="answer-not-found-checkbox"></input>  Answer not found? <br />
          </div>-->
          <div id="annotations">
          </div>
          <div id="buttons">
            <p id="pay-hint" class="small-hint"></p>
            <div id="warning-box-input-type-1" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-2" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-3" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-4" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-5" class="warning-box" style="display:none"></div>
            <p id="validated-hint" class="small-hint red"></p>
            <div id="guide-box-input-type" class="guide-box" style="display:none"></div>
          </div>
          <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
          <br />
          <br />
          <input type="checkbox" class="custom-control-input" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
        </div>
      </div>
    </div>`);
  }


  function loadQualificationHTML() {
    $('#taskContent').html(
      `<div class="container" id="container" role="main"><div class="panel panel-default">
        <div class="panel-heading"><button id="instruction-header" type="button" class="btn-lg" ></button></div>
        <div class="panel-body" id="instruction-body">
          <nav class="navbar navbar-default">
            <div class="container-fluid">
              <ul class="nav navbar-nav">
                <li class="active"><a href="#" id="instructions-overview" class="instructions-item">Overview</a></li>
                <li><a href="#" id="instructions-step-by-step" class="instructions-item">Step-by-step instructions</a></li>
                <li><a href="#" id="instructions-examples" class="instructions-item">FAQ</a></li>
                <li><a href="#" id="instructions-bonuses" class="instructions-item">Bonuses</a></li>
              </ul>
            </div>
          </nav>
          <div id="instructions">
            Instructions (TODO)
          </div>
        </div>
      </div>
      <p><span class="green">Green</span> means your response is correct,
      and <span class="yellow">Yellow</span> means your response is incorrect.
      Note that you can get qualification with 8 correct responses, and can submit with 4+ correct responses.
      <strong>Note: you can only work on one HIT. Otherwise your submission may be rejected. </strong>
      </p>
      <div id="div-for-buttons"></div>
      <div class="row">
        <div class="col">
          <div class="panel panel-default narrow-panel">
            <div class="panel-heading">Input Question</div>
            <div class="panel-body" id="input-question">(Loading...)</div>
          </div>
          <!-- Input box for User Input -->
          <div class="input-group narrow-input-group" id="write-question">
            <input class="editOption editable" id="search-query" placeholder="Write Query for Search Wikipedia" />
            <span class="input-group-addon btn btn-default run" id="search-button">Search</span>
            <br />
          </div>
          <div id="search-results"></div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-search-results" style="display:none;">< Go back to search results</button>
          <div id="wikipedia-box" class="narrow-wikipedia-box"></div>
        </div>
        <div class="col side">
          <button type="button" class="btn btn-primary go-back" id="go-back-to-default-annotation">Go back to default input</button>
          <button type="button" class="btn btn-primary go-back" id="go-to-step1" style="display:none">Go back to step1</button>
          <button type="button" class="btn btn-primary go-back" id="go-to-step2">Go to step2</button>
          <button type="button" class="btn btn-danger" id="hint-for-qualification" style="display:none">I need hints!</button>
          <button type="button" class="btn btn-success" id="check-my-response">Check my response</button>
          <p id="description-of-step1">
            Step 1: combine, delete, separate, or correct the answer.
          </p>
          <p id="description-of-step2" style="display:none">
            Step 2: lightly edit or add questions to allow a single clear answer; check questions to exclude.
          </p>
          <!--<div id="checkboxes">
            <input type="checkbox" class="custom-control-input" id="single-clear-answer-checkbox"></input>  Single clear answer? <br />
            <input type="checkbox" class="custom-control-input" id="answer-not-found-checkbox"></input>  Answer not found? <br />
          </div>-->
          <div id="annotations">
          </div>
          <div id="buttons">
            <p id="pay-hint" class="small-hint"></p>
            <div id="warning-box-input-type-1" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-2" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-3" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-4" class="warning-box" style="display:none"></div>
            <div id="warning-box-input-type-5" class="warning-box" style="display:none"></div>
            <p id="validated-hint" class="small-hint red"></p>
            <div id="guide-box-input-type" class="guide-box" style="display:none"></div>
          </div>
          <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
          <br />
          <br />
          <input type="checkbox" class="custom-control-input" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
        </div>
      </div>
    </div>`);
  }


  function loadFinalHTML() {
    $('#taskContent').html(
      `
      <div class="container" id="container" role="main">
      <div class="row">
        <div class="col">
          <div class="panel panel-default narrow-panel">
            <div class="panel-heading">Input Question</div>
            <div class="panel-body" id="input-question">(Loading...)</div>
          </div>
          <!-- Input box for User Input -->
          <div class="input-group narrow-input-group" id="write-question">
            <input class="editOption editable" id="search-query" placeholder="Write Query for Search Wikipedia" />
            <span class="input-group-addon btn btn-default run" id="search-button">Search</span>
            <br />
          </div>
          <div id="search-results"></div>
          <button type="button" class="btn btn-primary go-back" id="go-back-to-search-results" style="display:none;">< Go back to search results</button>
          <div id="wikipedia-box" class="narrow-wikipedia-box"></div>
        </div>
        <div class="col side">
          <div id="promptAnnotationsDiv"></div>
          <button type="button" class="btn btn-primary" id="submit">Submit annotations</button>
          <br />
          <textarea placeholder="Feedback (Optional)" class="" rows="4" id="feedback" name="feedback"></textarea>
          <input type="checkbox" class="custom-control-input" id="uw-checkbox"></input>  <span class="small">Are you an employee of the UW, family member of a UW employee, or UW student involved in this particular research?</span><br />
        </div>
      </div>
    </div>`);
  }


  function getGenerationInstructions() {
    return {
      'instructions-overview': `<p><span class="red bold">
      Please read the instructions thoroughly before beginning.
      Full understanding of the instructions will help you to get bonuses
      and obtain qualification to the batches with higher pay.
    </span></p>
    <p>
      <span class="bold">
        The batches with higher pay will be available until March.
        Qualifications to the batches with higher pay require you to work 40 hrs a week
        only on our batches. If you cannot work 40 hrs a week in March, we recommend
        not to work on our HITs.
      </span>
      <br />
      <span class="bold red">
        You can obtain qualifications to the batches with higher pay if:
      </span>
      <ol>
        <li>You have submitted <span class="bold red">more than 50 HITs</span>.</li>
        <li>Your accuracy and coverage is high enough (correct responses in 90% cases).</li>
        <li>You submitted sufficient number of HITs with multiple question-answer pairs.
        This means you should not submit too many 'single answer' or 'no answer' cases.</li>
      </ol>
    </p>

    <hr /><hr />

    <p>
      The goal of this task is to find all possible intended question-answer pairs given a prompt question, using the provided search engine. This is a two step process:
      <ol>
        <li>
          Given a prompt question, find all possibly-intended answers, using the provided search engine to navigate Wikipedia.
        </li>
        <li>
          (If there are multiple answers) For each possible answer, refine the prompt question to be more specific so that it only allows this answer. (We will call it intended question-answer pair.)
        </li>
      </ol>

      <strong>Definition:</strong> <em><strong>prompt question</strong></em> indicates the given question on the left side (with bigger font) that potentially contains ambiguity,
      and <em><strong>written questions</strong></em> indicate the questions on the right side
      that you will write, which should not contain any ambiguity.
      If it just says <em>questions</em>, it is likely to be referring to the written questions, not the prompt question.
      <em><strong>Answers</strong></em> indicate answers to each written question.
      Pairs of a written question and its answer together will be your <em><strong>response</strong></em>.
      <hr />

      Consider a prompt question <span class="q">When is hotel transylvania 3 going to come out?</span>,
      you will probably search for the article <a href="https://en.wikipedia.org/wiki/Hotel_Transylvania_3:_Summer_Vacation" target="_blank">Hotel Transylvania 3: Summer Vacation</a>
      and find that the answer can be <span class="q">June 13, 2018</span> or <span class="q">July 13, 2018</span>,
      depending on whether the film festival release is considered or only the theatrical release is considered.
      Therefore the output will be:
      <br />
      <img src="` + imgURLPrefix + `/movieRelease.png" width="550px" /><br />
    </p>
    <br />
    <p>
      Often, multiple intentions can only be found when you read Wikipedia extensively
      to find the answer. For example, the question about Hotel Transylvania 3 may seems clear,
      before actually reading <a href="https://en.wikipedia.org/wiki/Hotel_Transylvania_3:_Summer_Vacation" target="_blank">Hotel Transylvania 3: Summer Vacation</a>
      and finding both the film festival release date or the theatrical release date.
      Therefore, you should read multiple Wikipedia articles very carefully (8-10 min).
      We encourage you to use <span class="bold">CTRL+F / CMD+F</span> for navigating the document using keywords.
    </p>
    <p>
      Now, each prompt question falls into one of four categories (see "Examples" for full understanding):
      <ul>
        <li>
          <span class="bold">[Standard case]</span> all possible intended question-answer pairs should be written.
        </li>
        <li>
          <span class="bold">[Time-dependent case]</span> if the answer depends on when the question is asked,
          question-answer pairs should be refined to eliminate the time dependency,
          with up to three specific time-based variants, from December 31, 2017 and going backward.
          e.g. <span class="q">What date was Thanksgiving last year?</span> will be refined into 3 questions,
          <span class="q">What date was Thanksgiving in 2017?</span>,
          <span class="q">What date was Thanksgiving in 2016?</span>, and
          <span class="q">What date was Thanksgiving in 2015?</span>
        </li>
        <li>
          <span class="bold">[Single clear answer]</span> if you can find only one answer even after extensive search, mark “Single clear answer”
        </li>
        <li>
          <span class="bold">[Answer not found]</span> if the answer cannot be found from Wikipedia or the answer cannot be expressed in a short form text, mark “Answer not found”
        </li>
      </ul>
      Usually, we have around 35% of [Standard/Time-dependent], 60% of [Single clear answer], and 5% of [Answer not found].)
      <span class="red">If you have too many [Single clear answer]s or [Answer not found]s,
      you may be disqualified.</span>
    </p>
    <p>
      In order to maintain accuracy and coverage,
      each prompt question should take around <span class="large hl">8-10 minutes</span>.
    </p>
    <p>
      <span class="bold large">Important Notes!!!</span>
    </p>
    <p>
      <span class="bold">[Answer]</span> The answer should be a phrase of one or multiple
      (usually less than 8) continuous words from Wikipedia page you found (text, table or infobox).
      Please use the shortest possible text span, exactly as it appears in the sentence that
      supports the answer to the question.
      No explanation, no justification.
      Importantly, if there are multiple text expressions for the single answer,
      please write all of them, separated by <span class="q">|</span>, as comprehensively as possible.
      <ul>
        <li>
          Example: If the question is asking about the release date, write <span class="q">June 13, 2018</span>
          instead of <span class="q">It was released on June 13, 2018</span>.
        </li>
        <li>
          Example: If the question is asking about the budget of the film, and it was estimated at $275 million,
          write <span class="q">estimated at $275 million|$275 million</span>.
        </li>
      </ul>
    </p>
    <p>
      <span class="bold">[Question]</span> Written questions should be as close as possible
      to the prompt question, only edited to the minimal extend necessary when refining the question
      to differentiate between multiple intentions.
      This should be true even when the given question is ungrammatical or incomplete.
      Each written question should be standalone and does not depend on other written questions.
      Note that you should not write all questions related to the prompt question---you will write
      questions that can be one interpretation of the prompt question,
      and answers that can be the answers to the prompt question.
      <ul>
        <li>
          Example: <span class="q">Who won backstroke swimming in 2016 olympics?</span>
          <br />
          Wrong output (answers omitted):
          <blockquote>
            Who is the winner of 100m backstroke swimming in 2016 olympics?<br />
            Who is the winner of 200m backstroke swimming in 2016 olympics?<br />
            Which country is the winner of 100m backstroke swimming in 2016 olympics?<br />
            Which country is the winner of 200m backstroke swimming in 2016 olympics?
          </blockquote>
          Correct output (answers omitted):
          <blockquote>
            Who won 100m backstroke swimming in 2016 olympics?<br />
            Who won 200m backstroke swimming in 2016 olympics?<br />
            Which country won 100m backstroke swimming in 2016 olympics?<br />
            Which country won 200m backstroke swimming in 2016 olympics?
          </blockquote>
        </li>
      </ul>
    </p>
    <p>
      Note that the main goal is to consider <span class="red bold">what the user who asked the question
      might have actually wanted to know when they asked the question</span>,
      rather than what is the most likely answer, or the literal meaning of
      the language in question.
      (e.g. In the above question, <span class="q">Who won backstroke swimming in 2016 olympics?</span>,
      the question is asking about "who", but it is possible that the user meant the country.)
      However, do not write the question that cannot be intended (e.g. <span class="q">What is the record for backstroke swimming in 2016 olympics?</span>
      is certainly not an intended question to the prompt question.)
    </p>
    <hr /><hr />
    <p>
      Now, see <span class="bold">Examples</span> to see the more examples for each case.
      Reading each of them carefully will greatly maximize the quality of your work.
      See <span class="bold">FAQ</span> shows frequent mistakes from workers.
      Reading them / Revisiting them when you are not sure while working on HITs
      will be very helpful to obtain qualifications.
    </p>
      `,
      'instructions-step-by-step': `
    <p>
      Here, we will show some examples for full understanding of each case.
    </p>
    <p>
      <span class="bold">[Standard case]</span> all possible intended question-answer pairs
      should be written.
      <ul>
        <li>
          Example: <span class="q">When did the mindy project move to hulu?</span>
          <br />
          <img src="` + imgURLPrefix + `/mindyProject.png" width="350px" />
        </li>
        <li>
          Example: <span class="q">When did daylight saving first start in Australia?</span>
          <br />
          <img src="` + imgURLPrefix + `/daylightSaving.png" width="470px" />
        </li>
        <li>
          Example: <span class="q">Who sang riding on the city of new orleans?</span>
          <br />
          <span class="red">Incorrect case 1</span>: The second question depends on the first question.
          <br />
          <img src="` + imgURLPrefix + `/newOrleansIn1.png" width="400px" />
          <br />
          <span class="red">Incorrect case 2</span>: The questions have too many edits than necessary.
          <br />
          <img src="` + imgURLPrefix + `/newOrleansIn2.png" width="400px" />
          <br />
          <span class="bold">Correct case</span>: The questions are close to the question while differentiating multiple possible intents.
          <br />
          <img src="` + imgURLPrefix + `/newOrleans.png" width="400px" />
        </li>
      </ul>
    </p>
    <p>
      <span class="bold">[Time-dependent case]</span> if the answer depends on
      when the question is asked, question-answer pairs should be refined to eliminate
      the time dependency, with up to three specific time-based variants,
      from December 31, 2017 and going backward.
      <ul>
        <li>
          Example: <span class="q">When does next episode of dragon ball super air?</span>
          <br />
          <img src="` + imgURLPrefix + `/dragonBallSuper.png" width="350px" />
        </li>
        <li>
          Here, the question can contain two different intentions, and each of them is time-dependent.
        </li>
      </ul>
    </p>
    <p>
      <span class="bold">[Single clear answer]</span> if you can find only one answer
      even after extensive search, mark “Single clear answer”. Don't forget writing
      all possible text expressions.
      <ul>
        <li>
          Example: <span class="q">Author of book orange is the new black?</span>
          <br />
          <img src="` + imgURLPrefix + `/bookAuthor.png" width="350px" />
        </li>
        <li>
          Example: <span class="q">Which team has the most ncaa tournament appearances?</span>
          <br />
          <img src="` + imgURLPrefix + `/kentucky.png" width="470px" />
        </li>
        <li>
          Here, the question is not grammatical, but there is only one possible intention.
        </li>
      </ul>
    </p>
    <p>
      <span class="bold">[Answer not found]</span> if the answer cannot be found from Wikipedia or
      the answer cannot be expressed in a short form text, mark “Answer not found”
      <ul>
        <li>
          Example: <span class="q">When can an employee's religious belief qualify as a bona fide occupational qualification?</span>
          <br />
          <img src="` + imgURLPrefix + `/answerNotFound.png" width="150px" />
        </li>
        <li>
          Here, the answer can be found in Wikipedia, but it cannot be expressed in short form text.
        </li>
      </ul>
    </p>
    <p>
      Note that, when considering all possibilities, rather than considering the literal meaning of the language in question,
      please consider what would be the users' <span class="red bold">intent</span> when asking the question.
    </p>
      `,
      'instructions-examples': `
        <span class="bold">FAQ</span>
        <ol class="sep">
          <li>
            I am not sure what do you mean by "the written questions should be close to the prompt question
            but editted to differentiate multiple possible intents." ` + ARROW + `
            Go to [Examples] and see the examples of [Multiple possible question-answer pairs: standard].
            For example, if the prompt question is <span class="q">Who sang riding on the city of new orleans?</span>,
            and you want to describe the song in 1984,
            you might think both <span class="q">Who is the singer of the riding on the city of new orleans in 1984?</span>
            and <span class="q">Who sang riding on the city of new orleans in 1984?</span> are both correct.
            However, you need to write the latter, not the former, in order to keep the prompt question as much as possible.
            Note that the goal of this task is to remove ambiguity of the prompt question, so the edits from the prompt question should be
            only for disambiguation.
          </li>
          <li>
            Prompt question looks ambiguous but I can only find one intention.
            ` + ARROW + ` Mark "Single clear answer" (See Example 1).
            <br /><br />
            Example: <span class="q">The day sweden changed from left to right?</span>
            <br />
            Response: <span class="q">3 September 1967</span>
          </li>
          <li>
            Prompt question has "where" but not sure if it means a city or a country.
            ` + ARROW + ` Do not separate them but include all of them as all possible text spans,
            as they do not really mean different intentions.
            It is good to write the most specific place as much as possible
            (this applies for the year/date).
            e.g. <span class="q">Vancouver, Canada|Vancouver|Vancouver, BC, Canada</span>
          </li>
          <li>
            Prompt question is clearly ambiguous, but I can only find one intended question that
            has the answer, and I cannot find the answer for other possible intended questions.
            ` + ARROW + ` Mark "Single clear answer", as we only consider questions
            with the answer.
          </li>
          <li>
            Prompt question asks about two different things in the same time.
            e.g. <span class="q">When was the hubble space telescope launched and
            when will it stop operating?</span> ` + ARROW + ` Separate them into
            two questions, <span class="q">When was the hubble space telescope launched?</span>
            and <span class="q">When will the hubble telescope stop operating?</span>
          </li>
          <li>
            The question is too general. ` + ARROW + ` We recommend to select "no answer".
            e.g.
            <span class="q">The cast of a good day to die hard?</span>,
            <span class="q">Who starred in the movie summer of 42?</span>, or
            <span class="q">What are the band members names of the rolling stones?</span>
          </li>
          <li>
            When the question is about the release date of the "movie",
            should we include DBD or Blu-ray release? ` + ARROW + `
            Please do not include DVD or blu-ray release, as it is not the first release. Here is the example of what should be included and what should not.
            <br />
            Question: <span class="q">When does the fifty shades of grey come out?</span>
            <br />
            Question-answer pairs to be included:
            <br />
            <span class="q">
              Q: When did the book Fifty Shades of Grey come out? / A: June 20, 2011
            </span>
            <br />
            <span class="q">
              Q: When did the movie Fifty Shades of Grey come out in Los angeles? / A: February 9, 2015
            </span>
            <br />
            <span class="q">
              Q: When did the movie Fifty Shades of Grey come out all over the US? / A: February 13, 2015
            </span>
            <br />
            Question-answer pair not to be included:
            <br />
            <span class="q">
              Q: When does the film Fifty Shades of Grey come out in DVD and Blu-ray? / A: May 8, 2015
            </span>
          </li>
          <li>
            When the question is about the release date of the “song”, should we include cover songs? ` + ARROW + `
            In many cases please do not include the cover songs if they are unofficial.
            It is OK to include the successor songs that are official.
          </li>
          <li>
            If the question is asking about the movie release date and there are 5 different movies with the same title,
            should we include the latest 3 movies, assuming it is time dependent? ` + ARROW + `
            No. You should include all of them.
            Time-dependency is usually for events that occur regularly in a time basis (e.g. election, sports game, etc).
          </li>
        </ol>
      `,
      'instructions-bonuses': `
<p>
Aside from the base pay $0.2, you will get bonus when your response pass the validation.
In order to pass the validation, your response should (1) contain all correct answers, and
(2) cover all possible intents.
</p>

<ul>
  <li>
    When your response contains a single clear answer, you get +$0.1 bonus
    <span class="bold red">(total $0.3)</span>
  </li>
  <li>
    When your response multiple question-answer pairs, if all pairs pass validations and no other validated pairs found,
    you get $0.4 (with 2 pairs) / $0.5 (with 3+ pairs) bonus
    <span class="bold red">(total $0.6-0.7)</span>
  </li>
  </ul>
<span class="red">
  Note that, if you obtain qualifications, the batches you will work on
  are guanranteed to pay $480 a week (and usually more) and give at least $0.75/HIT.
  We highly recommend to work on these HITs only when your goal is to
  obtain qualifications.
</span>
`
    };
  }

  function getValidationInstructions() {
    return {
      'instructions-overview': `<p><span class="large bold">
      Please read the instructions thoroughly before beginning. Full understanding of the instructions
      will help you to retain your qualification and get <span class="hl">bonuses
      (up to 50cents per example; see Bonuses section)</span>.
    </span></p>

    <p>
      The goal of this task is to verify all possible intended-answers given a prompt question.
      You will be given a prompt question and a set of question-answer pairs,
      where each pair contains a possibly intended question and its answer.
      These question-answer pairs are from three other workers.
      <strong>Your job is to verify each of them and complete the most correct and comprehensive question-answer pairs.</strong>
    </p>

    <p>
      <strong><span class="large">Definition:</span></strong> <em><strong>prompt question</strong></em> indicates the given question on the left side (with bigger font) that potentially contains ambiguity,
      and <em><strong>written questions</strong></em> indicate the questions on the right side that you will assess (or edit), which should not contain any ambiguity.
      If it just says <em>questions</em>, it is likely to be referring to the written questions, not the prompt question.
      <em><strong>Answers</strong></em> indicate answers to each written question.
      Pairs of a written question and its answer together will be your <em><strong>response</strong></em>.
    </p>

    <p>
      This is a two step process:
    </p>

    <p>
      <strong>[Step 1]</strong>
      You will be given a prompt question and a set of answers (with questions that workers wrote, but you can hide them if they are distracting).
      You will complete a comprehensive set of possible answers by combining, deleting, separating or correcting them.
      <ul>
        <li>
          Combine answers from different boxes, in case they have the same meaning
          (e.g. <span class="q">U.S.</span> and <span class="q">the United States of America</span>).
        </li><li>
          Separate answers in the same form, in case they are from different intents
          (e.g. <span class="q">Ryan Murphy</span> is the answer for both
          <span class="q">Who is the winner of 100m backstroke swimming in 2016 olympics?</span> and
          <span class="q">Who is the winner of 200m backstroke swimming in 2016 olympics?</span>,
          but they belong to different intentions).
        </li><li>
Delete answers if they are completely wrong.
</li><li>
If you want to correct the answer instead of deleting them, edit the text boxes.
</li>
</ul>
</p>

<p>
<strong>[Step 2]</strong>
You will be given questions from workers for each answer.
Each question should (i) allow only the paired answer, and
(ii) be refined from the prompt question to the minimal extent to differentiate multiple intentions.
If you feel the question totally doesn’t make sense, exclude it by checking the checkbox.
If the question makes sense but does not perfectly satisfy the requirement, edit it.
</p>

<p>
Note that each step requires <span class="strong">searching and reading of Wikipedia pages</strong>, as it requires validating the answers
and writing the questions with a single clear answer.
You will use the provided search engine to navigate Wikipedia.
</p>

<p>
This is the end of the overview. You will have to read the step-by-step instructions for full understanding.
</p>
<p>
<span class="large red">
Below are important notes before you move on to the step-by-step instructions.
</span>
</p>
<p>
Remember that each prompt question should fall into one of four categories:
<ul>
<li>
<strong>[Standard case]</strong> all possible intended question-answer pairs should be written.
</li><li>
<strong>[Time-dependent case]</strong> if the answer depends on when the question is asked,
question-answer pairs should be refined to eliminate the time dependency,
with up to three specific time-based variants, from Dec 2017 going backwards
(e.g. <span class="q">What date was Thanksgiving last year?</span> will be refined into 3 questions,
<span class="q">What date was Thanksgiving in 2017</span>,
<span class="q">What date was Thanksgiving in 2016</span> and
<span class="q">What date was Thanksgiving in 2015</span>.
</li><li>
<strong>[Single clear answer]</strong> it is possible that you find only one answer to be correct.
In this case, you will skip Step 2.
</li><li>
<strong>[Answer not found]</strong> it is possible that you find no answer to be correct,
or the answer to be too long (e.g. entire paragraph).
Then, you will delete all answers to indicate that there is no (short) answer.
In this case, you will skip Step 2 as well.
</li>
</ul>
</p>

<p>
<strong>[Answer]</strong> The answer should be a phrase of one or multiple (usually less than 8) continuous words from Wikipedia page
you found (text, table or infobox). Please use the most shortest text span, exactly as it appears in the sentence that supports the answer to the question.
No explanation, no justification. Importantly, if there are multiple text expressions for the single answer,
we want all of them should be included, separated by <span class="q">|</span>, as comprehensively as possible.
<ul>
<li>
Example: If the question is asking about the release date, the answer should be <span class="q">June 13, 2018</span> instead of
<span class="q">It was released on June 13, 2018</span>.
</li>
</ul>
</p>

<p>
<strong>[Question]</strong> Written questions should be as close as possible to
the prompt question (the question shown in the left side with a bigger font),
only edited to the minimal extent necessary when refining the question to differentiate between multiple intentions.
This should be true even when the given question is ungrammatical or incomplete.
Each written question should be standalone and does not depend on other written questions.
Note that you should not include all questions related to the prompt question---you will include questions
that can be one interpretation of the prompt question, and answers that can be the answers to the prompt question.
</p>


<p><span class="red large">TIPS!</span></p>
<p>
We encourage you to use CTRL+F / CMD+F for navigating the document using keywords.
More than 30% of prompt questions should have multiple question-answer pairs.
In less than 5% of cases, you will choose all answers are invalid.
In other cases, you will end up with a single answer (potentially multiple text expressions.)
If your statistics are very different from these statistics,
you might have misunderstood the task, so please review all instructions and examples more carefully.
</p>

<p><span class="bold large">Now, move on to the step-by-step instructions for the full details.</span></p>
      `,
      'instructions-step-by-step': `

<p><span class="strong large">Step 1</span></p>

<p>
You will be given a set of answers (with questions that workers wrote, but you can hide them if they are distracting).
You will complete a comprehensive set of possible answers by combining, deleting, separating or correcting them.
</p>

<ol>
<li>
Combine answers from different boxes, in case they have the same meaning, by checking the checkboxes and
click <span class="button-span">Combine</span> button.
<br />
Example: <span class="q">Author of book orange is the new black?</span>
<br />
<img src="` + imgURLPrefix + `/author-1-in.png" width="350px" /><br />
<br />
<img src="` + imgURLPrefix + `/author-out.png" width="350px" /><br />
</li>
<li>
Separate answers in the same form, in case they are from different intents, by checking the checkbox and
click <span class="button-span">Separate</span> button.
<br />
Example: <span class="q">Who won backstroke swimming in 2016 olympics?</span>
<br />
<img src="` + imgURLPrefix + `/swimming-in.png" width="450px" /><br />
<br />
<img src="` + imgURLPrefix + `/swimming-out.png" width="350px" /><br />
</li>
<li>
Delete answers if they are completely wrong, by checking the checkboxes and click <span class="button-span">Delete</span> button.
<br />
Example: <span class="q">Author of book orange is the new black?</span>
<br />
<img src="` + imgURLPrefix + `/author-2-in.png" width="350px" /><br />
<br />
<img src="` + imgURLPrefix + `/author-out.png" width="350px" /><br />
</li>
<li>
If you want to correct the answer instead of deleting them, edit the text boxes.
<br />
Example: <span class="q">How much budget was used for solo: a star wars story?</span>
<br />
<img src="` + imgURLPrefix + `/long-answer-in.png" width="450px" /><br />
<br />
<img src="` + imgURLPrefix + `/long-answer-out.png" width="450px" /><br />
</li>
</ol>

<p><span class="strong large">Step 2</span></p>
<p>
You will be given questions from workers for each answer.
Each question should (i) allow only the paired answer, and
(ii) be refined from the prompt question to the minimal extent to differentiate multiple intentions.
Note that if you ended up with 1 or 0 answer in Step 1, you will skip Step 2, as there is no multiple intentions to differentiate.

You have three things to do.
<ol>
<li>
If the question is blank, write a question that meets the above requirement.
</li><li>
If you see the question that doesn’t make sense, exclude it by checking the checkbox.
</li><li>
If the question makes sense but does not perfectly satisfy the requirement, edit it.
</li>
</ol>
Now, let's take a look at examples to see how it works.

<ul>
<li>
Example: <span class="q">When did daylight saving start in austrailia?</span>
<br />
<img src="` + imgURLPrefix + `/daylight-saving-in.png" width="400px" /><br />
Here you can write a question for the first question.
But then you realize the second & third questions should be refined
not to allow the answer <span class="q">during world war I</span>.
So they will also be refined.
<br />
<img src="` + imgURLPrefix + `/daylight-saving-out.png" width="400px" /><br />
</li><li>
Example: <span class="q">Who sang riding on the city of new orleans?</span>
<br />
<img src="` + imgURLPrefix + `/new-orleans-1-in.png" width="400px" /><br />
All of questions kinda make sense, but the second question is not so clear by standalone.
Also, they have unnecessary edits from the prompt question.
So you will edit them to have words copied from the prompt question as much as possible.
<br />
<img src="` + imgURLPrefix + `/new-orleans-1-out.png" width="400px" /><br />
</li><li>
Example: <span class="q">Who sang riding on the city of new orleans?</span>
<br />
<img src="` + imgURLPrefix + `/new-orleans-2-in.png" width="400px" /><br />
First of all, the first question for <span class="q">Arlo Buthrie</span>
should be excluded as it cannot be standalone.
Other questions can be lightly editted to be similar to the prompt question.
<br />
<img src="` + imgURLPrefix + `/new-orleans-2-out.png" width="400px" /><br />
Note that we exclude the second question for <span class="q">Steve Goodman</span>
but it is possible to include it and edit it to be similar to the prompt question.
</li>
</ul>

Note that you can see red error messages or yellow warning messages during Step 2.
Error messages are crucial and you need to resolve them to submit your answer.
Warning messages are just warning based on our algorithms, just to point out some common mistakes - you can ignore them.

</p>

<span class="red bold">TIPS!!</span>
<br />
<ul>
<li>
  This task can be really subjective. When you are working, don't be frustrated if you are not sure what to do!
  Unless you have a reasonable choice within the scope of instructions, they will be acceptable.
  For qualification, we allow multiple options to accept any reasonable cases, to help you getting a sense of what are *reasonable*.
  In practice, we will evaluate your HIT based on agreement with a threshold
  adjusted based on the uncertainty of the task.
</li>
<li>
Be generous. Do not remove the answer or the question unless they are totally nonsense
or clearly violates requirements.
</li>
<li>
For the answer, including possible text expressions as many as possible will greatly increase the probability of agreement and bonuses.
</li>
<li>
When reviewing your response, we do not consider punctuations, capitalization and whitespaces,
so you can ignore them.
</li>
</ul>
</p>
<p>
  Here are three videos with the actual process (note that the video is a bit fast :)).
</p>
<video width="800" controls>
  <source src="` + imgURLPrefix +  `/nfl.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
<video width="800" controls>
  <source src="` + imgURLPrefix +  `/new_edition.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
<video width="800" controls>
  <source src="` + imgURLPrefix +  `/general_hospital.mp4" type="video/mp4">
  Your browser does not support HTML5 video.
</video>
      `,
      'instructions-examples': `
<ol>
<li>
Prompt question asks about two different things in the same time.
e.g. <span class="q">When was the hubble space telescope launched and when will it stop operating?</span>
` + ARROW + `Separate them into two questions,
<span class="q">When was the hubble space telescope launched?</span> and
<span class="q">When will the hubble telescope stop operating?</span>
</li>
<li>
I am not sure what do you mean by "the written questions should be close to the prompt question
but editted to differentiate multiple possible intents." ` + ARROW + `
For example, if the prompt question is <span class="q">Who sang riding on the city of new orleans?</span>,
and you want to describe the song in 1984,
you might think both <span class="q">Who is the singer of the riding on the city of new orleans in 1984?</span>
and <span class="q">Who sang riding on the city of new orleans in 1984?</span> are both correct.
However, you need to write the latter, not the former, in order to keep the prompt question as much as possible.
Note that the goal of this task is to remove ambiguity of the prompt question, so the edits from the prompt question should be
only for disambiguation.
</li>
<li>
I ended up with one answer because I cannot find the other possible answer.
However, I still feel like the prompt question is clearly ambiguous.
What should I do?` + ARROW + `No need to worry. It is fine to have only one answer and skip Step 2.
For example, <span class="q">The day sweden changed from left to right?</span> looks ambiguous.
However, when you try to find the answer, you realize that the only possible intent of this question is
about when did the traffic in Sweden switch from driving on the left-hand side of the road to the right.
</li>
<li>
I have no idea what you mean by "if the answer depends on when the question is asked,
question-answer pairs should be refined to eliminate the time dependency,
with up to three specific time-based variants, from Dec 2017 going backwards". ` + ARROW + `
We totally understand that it sounds confusing, but once you understand it, it is a really easy case.
Consider this example: <span class="q">Where is the Winter Olympics held?</span>
The answer depends on when the question is asked.
Imagine it was asked in Dec 2017. Then it will be asking about 2018 Winter Olympics.
Now, imagine it was asked before that. It might be asking about 2014 Winter Olympics, 2010 Winter Olympics and so on.
As the instructions says "up to three specific time-based variants", the final question-answer pairs should look like this.
<br />
<img src="` + imgURLPrefix + `/winter-olympics-final.png" width="400px" />
<br />
Note that having comprehensive answers like this is not required, but it can increase the agreement ratio and bonuses.
</li>
<li>
I am really not sure what to do for this HIT ` + ARROW + ` You can delete
all answers, skip Step 2 and submit the HIT.
However, if you do it so often, you will be disqualified.
</li>
</ol>

      `,
      'instructions-bonuses': `
<p>
Aside from the base pay 20c, you will get the bonuses based on number of the given answers
and our manual validation. The manual validation will be done at the probability of 50%.
<ol>
<li>
  If you are given more than 4 answers to assess,
  <ul><li>
  If your response passes our validation,
  you will earn 30c bonus (<span class="red bold">50c in total</span>).
  </li><li>
  If your response skips validation,
  you will earn 20c bonus (<span class="red bold">40c in total</span>).
  </li></ul>
</li>
<li>
  If you are given more than 2 answers to assess,
  <ul><li>
  If your response passes our validation,
  you will earn 20c bonus (<span class="red bold">40c in total</span>).
  </li><li>
  If your response skips validation,
  you will earn 10c bonus (<span class="red bold">30c in total</span>).
  </li></ul>
</li>
<li>
  In other cases, if your response matches with other validators' response,
  you will earn 5c bonus (<span class="red bold">25c in total</span>).
</li>
</ol>
</p>
<p>
Matches between validators will be judged based on our internal algorithms as well
as human reviewers. (Note: They are more likely to be matched when you write
all possible span expressions of the answer as comprehensive as possible!)
</p>
<p><span class="large red">
When you maintain accuracy and coverage, you will earn much more than moving on to other questions quickly!
</span></p>
      `};
  }

  function updateQualificationInstructions(instructions) {
    instructions['instructions-overview'] = `
        <p><span class="red bold">
        You will be given 8 questions for qualification.
        We will automatically show if your response for each question was passed.
        If you cannot figure it out, you can always email us---we usually reply
        in a few minutes (at most 1 hr).
        If you have already participating in the same qualification task, do not submit HIT.
        Your HIT may be rejected.
        (Other related tasks (e.g. generating question-answer pairs) are fine.)
        </span></p>
      <p>
        <ul>
        <li>
          <span class="red">
            If you pass all 8 questions, you will get qualification and rewarded $1 bonus (in total $2).
          </span>
        </li><li>
          <span class="red">
            If you pass 7 questions and the last response is reasonable,
            submit your HIT. We will review your response and give bonus & qualification
            if the response is reasonable (no need to email us to request review).
          </span>
        </li><li>
          <span class="red">
            If you pass 4--6 questions and think the other responses are also reasonable,
            submit your HIT and email us with your Worker ID so that we can review your HIT.
            Then you will also get $1 bonus (in total $2) and qualification.
          </span>
        </li><li>
          <span class="red">
            In other cases, you cannot get qualification.
            But you can always give up and submit to get a base pay, if you pass 4+ questions.
            We also encourage to email us if you want more hints.
          </span>
        </li></ul>
        <span class="red">
          Please do note, that if you do not pass at least 4 questions, you cannot submit the HIT.
        </span>
      </p>
      <p><span class="red bold">
        Once you are qualified, you will join our Slack to see schedules for HITs (20,000 HITs coming!).
        We expect the next main task will be February 10 (Mon), so stay tuned!
      </span></p>
      <p><span class="red bold">
        From now on, the instructions are exactly same as those for the main task.
      </span></p>` + instructions["instructions-overview"];
    instructions['instructions-examples'] = instructions['instructions-examples'] + `
      <p>
      <span class="large bold">Only for qualifications:</span>
      <ul>
        <li>
          I cannot figure it out even after the hints. ` + ARROW + `Send email to me.
          We don't want you to spend too much time on it. We will help you passing the qualification!
        </li>
        <li>
          I accidentally submitted the HIT before completing all 8 questions. What should I do? ` + ARROW + `
          Visit <a target="_blank" href="http://qa.cs.washington.edu:7777/task/qualification/preview#">this link</a> to see our interface
          and work on questions that you have not finished. When you see green buttons, send me your worker ID and a screenshot.
          Then we will give you bonus and qualification. You don't need to repeat questions that you have already completed - only work on
          questions you haven't finished and send us. Also, be careful not to click "Submit" button with this inference as it won't work.
        </li>
        <li>
          I want to take a break and try again, but I am worried about timeout. ` + ARROW + `
          If you have done at least 4 questions, submit your HIT, and,
          similar to above, visit <a target="_blank" href="http://qa.cs.washington.edu:7777/task/qualification/preview#">this link</a>
          to work on questions and send me a screenshot of green buttons.
          If you have not done at least 4 questions so cannot submit the HIT,
          save a screenshot of green buttons, and then revisit our HIT (or go to the url above)
          to complete the remaining questions. Then you can email me with your worker ID and a screenshot.
        </li>
        <li>
          I have returned the HIT. Am I not allowed to work on this HIT? ` + ARROW + `
          As long as you have not submitted it, you can work on it!
        </li>
        <li>
          I completed 4-7 questions, but I want to finish remaining questions. ` + ARROW + `
          Again, visit the above url and send me your worker ID and a screenshot.
        </li>
        <li>
          I think all of my responses are reasonable, but I cannot pass. ` + ARROW + `
          If you have done at least 4 questions, submit your HIT and simply send me your worker ID.
          Then I will review your HIT. If you have not done 4 questions,
          send me screenshots of your responses so that I can review them.
        </li>
      </ul>
      <p>
        If you are on <span class="bold">Turker Nation</span> slack,
        DM me (Sewon Min) for any questions or any hints.
      </p>
    `;
    return instructions
  }
})();


