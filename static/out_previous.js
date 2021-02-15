(function() {

  /* scripts for generations */

  var question_id;
  var prompt_question;
  var PAY_BASE = 10;
  var PAY_DELTA = 10;
  var PAY_MULTIQA = 20;
  var currentPay = 0;
  var cache = [];
  var imgURLPrefix = "https://shmsw25.github.io/assets/screenshots/"

  /* main code start */

  var instructions;
  $( window ).init(function(){
    if ($('#taskKey').attr("value")==='"final"') {
      loadFinalHTML();
      setWidth();
      loadAll(null);
      var allHTMLText = ""
      var promptAnnotations = JSON.parse($('#prompt').attr("value"))['annotations'];
      for (i in promptAnnotations) {
        let promptAnnotation = convertAnnotation(promptAnnotations[i]);
        var htmlText = ""
        if (promptAnnotation['type']==='noAnswer') {
          htmlText += "No answer found";
        } else if (promptAnnotation['type']==='singleAnswer') {
          htmlText += "Single clear answer<br /><span class='graybackground'>A:</span> ";
          htmlText += promptAnnotation['answer'];
        } else if (promptAnnotation['type']==='multipleQAs') {
          htmlText += "Multiple question-answer pairs</p>";
          promptAnnotation['qaPairs'].forEach( d => {
            htmlText += "<p><span class='graybackground'>Q:</span> " + d['question'] + "<br />";
            htmlText += "<span class='graybackground'>A:</span> " + d['answer'] + "</p>";
          } );
        }
        allHTMLText += `
          <div class="panel panel-default panel-inline">
            <div class="panel-heading">Option #` + (parseInt(i)+1).toString() + `
              <input type="checkbox" class="custom-control-input correct-checkbox" id="correct-checkbox-` + i + `"></input> Correct?<br />
            </div>
            <div class="panel-body" id="prompt-annotations">` + htmlText + `
            </div>
          </div>
          `;
      }
      $('#promptAnnotationsDiv').html(allHTMLText);
      $('.correct-checkbox').change(function () {
        var responses = []
        for (var i=0; i<promptAnnotations.length; i++) {
          responses.push(($('#correct-checkbox-'+i).prop('checked'))? 1 : 0);
        }
        $('#response').val(JSON.stringify(responses));
      });
      turkSetAssignmentID();
      return;
    }
    if ($('#taskKey').attr("value")==='"generation"') {
      instructions = getGenerationInstructions();
      loadGenerationHTML();
    } else if ($('#taskKey').attr("value")==='"validation"') {
      instructions = getValidationInstructions();
      loadValidationHTML();
    }
    setWidth();
    loadAll(instructions);
    if ($('#taskKey').attr("value")==='"validation"') {
      setPromptAnnotations();
      getAnnotations();
    } else {
      loadInitAnnotationForm();
    }
    turkSetAssignmentID();
	});

  function loadGenerationHTML() {
    $('#taskContent').html(
      HTML_PREFIX + `
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
      <div class="col col-6 col-md-4">
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
    $('#taskContent').html(
      HTML_PREFIX + `
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
        <div class="col col-6 col-md-4">
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
    </div>`);
  }

  function loadFinalHTML() {
    $('#taskContent').html(
      `
      <div class="container" id="container" role="main">
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
        <div class="col col-6 col-md-4">
          <div id="promptAnnotationsDiv"></div>
          <button type="submit" class="btn btn-primary" id="submit-button">Submit annotations</button>
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
      Please read the instructions thoroughly before beginning. Full understanding of the instructions
      will help you to retain your qualification and get <span class="large hl">bonuses (up to $1 per example; see Bonuses section)</span>.
    </span></p>
    <p>
      The goal of this task is to find all possible intended question-answer pairs given a prompt question,
      using the provided search engine.
      For instance, given the question <span class="q">When is hotel transylvania 3 going to come out?</span>,
      the output should be:
    </p>
    <img src="` + imgURLPrefix + `/movieRelease.png" width="550px" /><br />
    <br />
    <p>
      In order to be correct, the written question-answer pairs should meet all of the following criteria:
      <ol>
        <li>
          All possible <i>intended</i> question-answer pairs should be written. Written questions should have minimal edits from the prompt question.
        </li>
        <li>
          In case the answer is time-dependent, question-answer pairs should be
          refined to eliminate the time dependency, with up to three specific
          time-based variants. Targetted time should start from December 31, 2017 and go past.
          (please see details in 'step-by-step instructions'.)
        </li>
        <li>
          If there is only one clear answer, "Single clear answer" should be marked.
        </li>
        <li>
          If the answer cannot be found from Wikipedia, or the answer cannot be expressed in a short form text, "Answer not found" should be marked.
        </li>
      </ol>
    </p>
    <p>
      Often, multiple intentions can only be found when you actually attempt to find the answer.
      For example, the question about <i>Hotel Transylvania 3</i> may seems clear, before actually reading
      <a href="https://en.wikipedia.org/wiki/Hotel_Transylvania_3:_Summer_Vacation">Hotel Transylvania 3: Summer Vacation</a>
      and finding both the film festival release date or the theatrical release date.
      Therefore, you should read at least 3-5 Wikipedia articles very carefully (at least 8-10 minutes) to find out
      <span class="red bold">what the user who asked the question might have actually wanted to know when they asked the question</span>.
    </p>
    <p>
      Note that you get <span>the maximum bonus when your responses pass the validations and are comprehensive
      (no other pairs found)</span>. In order to maintain accuracy and coverage,
      each prompt question should take around <span class="large hl">8-10 minutes</span>. <span class="hl bold">You will earn much more than
      moving on to other questions quickly!</span>
    </p>
    <p>
      <span class="bold large">Important Notes!!! (Necessary to get bonuses; see step-by-step instructions and examples for full details)</span>
    </p>
    <p>
      <span class="bold red">Writing the answer:</span> The answer should be <span class="bold">a phrase of
      one or multiple (usually less than 10) continuous words</span> from Wikipedia page you found (text, table or infobox).
      Please use the minimal span without any explanation.
      Pleause use the answer span exactly as it appears in the sentence that supports the answer
      to the question.
      If there are multiple text expressions for the single answer, please write all of them,
      separated by <span class="q">|</span>, as comprehensively as possible.
      e.g. If the question is asking about the release date, write <span class="q">June 13, 2018</span> instead of
      <span class="q">It was released on June 13, 2018</span>.
      Or, if the question is asking about the budget of the film,
      write <span class="q">estimated at $275 million|$275 million</span>.
    </p>
    <p>
      <span class="bold red">Writing the question:</span>
      Written questions should be <span class="bold">as close as possible to the prompt question,
      only edited to the minimal extend necessary when refining the question to differentiate between multiple intentions</span>.
      This should be true even when the given question is ungrammatical or incomplete.
      Each written question should be standalone and does not depend on other written questions.
      Note that you should <span class="bold">not</span> write all questions related to the prompt question---you will write
      questions that can be one interpretation of the prompt question, and answers that
      can be the answers to the prompt question.
    </p>
    <p>
      <span class="bold large">TIPS!!!</span>
      <ul>
        <li>We encourage you to use CTRL+F / CMD+F for navigating the document using keywords.</li>
        <li>More than 30% of prompt questions should have multiple question-answer pairs, and less than 5% should be "answer not found".
        If your statistics are very different from these statistics, you might have misunderstood
        the task, so please review all instructions and examples more carefully.</li>
      </ul>
    </p>
    <p><span class="bold">
      See the Step-by-Step Instructions tab for full details on how to complete the task. <br />
    </span></p>
    <p><span class="red">
      We will invite workers with high accuracy and coverage to the larger batch in the near future.
    </span></p>
      `,
      'instructions-step-by-step': `
    <p>
      Step 1. Read the given question, and find one or multiple Wikipedia articles that is likely to contain the evidence to the question. Attempt to find all possible short form text that can be the answers.
    </p>
    <p>
      Step 2. Now, you will write the output according to the case out of 4 categories.
    </p>
    <p>
      <span class="bold">[Single clear answer]</span> There is a single clear answer to the given question, and no clarification required.
      <ul>
        <li>
          Read three or more Wikipedia documents to ensure that there is only one answer.
          If you are sure, check “Single clear answer” checkbox, and write an answer.
        </li>
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
      <span class="bold">[Multiple possible question-answer pairs: standard]</span> There are multiple possible interpretations to the given question, and the answers vary depending on it.
      <ul>
        <li>
          Write all possible intended question-answer pairs. Written questions should have minimal edits from the given question. Each question/answer should be different from each other.
        </li>
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
          Example: <span class="q">Who won backstroke swimming in 2016 olympics?</span>
          <br />
          <img src="` + imgURLPrefix + `/swimmingWinner.png" width="470px" />
        </li>
        <li>
          Here, although the answer text are the same, the question has two different possible intentions so that are considered as separate pairs.
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
      <span class="bold">[Multiple possible question-answer pairs: time-dependent]</span>
      The answer depends on when the question was asked.
      <ul>
        <li>
          Write question-answer pairs that are
          refined to eliminate the time dependency, with up to three specific
          time-based variants. Targetted time should start from December 31, 2017 and go past.
        </li>
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
      <span class="bold">[Answer not found]</span> Can't find the answer in Wikipedia, or answer cannot be expressed in a short form of the text.
      <ul>
        <li>
          Read three or more Wikipedia documents to ensure that there is no answer.
          If you are sure, check “Answer not found” checkbox.
        </li>
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
      It is very important to write <span class="red bold">all possible questions and answers as comprehensively as possible</span>.
      Required time for each question significantly varies based on the question, but
      <span class="red bold">minimum 5-10 minutes are required for questions with multiple question-answer pairs (and you will earn up to $1)</span>.
      Even though you are certain about one answer, please take more time to find other possibilities. (Often they are just harder to find!)
    </p>
    <p>
      Also note that, when considering all possibilities, rather than considering the literal meaning of the language in question,
      please consider what would be the users' <span class="red bold">intent</span> when asking the question.
    </p>
      `,
      'instructions-examples': `
        <span class="bold">FAQ</span>
        <ol class="sep">
          <li>
            I am not sure what do you mean by "the written questions should be close to the prompt question
            but editted to differentiate multiple possible intents." ` + ARROW + `
            Go to [Step-by-Step Instructions] and see the last example of [Multiple possible question-answer pairs: standard].
            It shows 2 incorrect cases and 1 correct case of the question <span class="q">Who sang riding on the city of new orleans?<span>.
          </li>
          <li>
            Prompt question looks ambiguous but I can only find one intention.
            ` + ARROW + ` Mark "Single clear answer" (See Example 1).
            <br /><br />
            Example: <span class="q">The day sweden changed from left to right?</span>
            <br />
            <img src="` + imgURLPrefix + `/sweden.png" width="470px" />
            <br />
          </li>
          <li>
            Prompt question has "where" but not sure if it means a city or a country.
            ` + ARROW + ` Mark "Single clear answer" and include all possible text spans,
            as they do not really mean different intentions (See Example 3).
            <br /><br />
            Example: <span class="q">Where is the Winter Olympics held?</span>
            <br />
            <img src="` + imgURLPrefix + `/winterOlympics.png" width="470px" />
            <br />
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
            There is an ambiguity in the question and the answer is time-dependent in the same time.
            ` + ARROW + `Consider all of them and write all possible pairs.
            For example, in the below example, 'Winner' can refer to 3 different things
            and the answer is time-dependent, so 9 question-answer pairs in total.
            (Although the lincoln horse race can refer to
            <a href="https://en.wikipedia.org/wiki/Lincoln_Heritage_Handicap">Lincoln Heritage Handicap</a>,
            there was no event for a long time. Therefore, we will only consider
            <a href="https://en.wikipedia.org/wiki/Lincoln_Handicap">Lincoln Handicap</a>.)
            <br /><br />
            Example: <span class="q">Who won the lincoln horse race last year?</span>
            <br />
            <img src="` + imgURLPrefix + `/horseRace1.png" width="470px" />
            <br />
            <img src="` + imgURLPrefix + `/horseRace2.png" width="470px" />
          </li>
        </ol>
      `,
      'instructions-bonuses': `
        <p>
          Aside from the base pay $0.1, you will get the bonuses based on validations by other crowdworkers.
        </p>
        <p>
          If you write a single clear answer <span class="red bold">(Maximum total $0.3)</span>
          <ul>
            <li>
              If answer passes validations and no other validated answers found: + $0.2 bonus
            </li>
            <li>
              If answer passes validations but other validated answers found: + $0.1 bonus
            </li>
            <li>
              If answer does not pass validations: 0 bonus
            </li>
          </ul>
        </p>
        <p>
          If you check “answer not found” <span class="red bold">(Maximum total $0.3)</span>
          <ul>
            <li>
              If no validated answers found: + $0.2 bonus
            </li>
            <li>
              If any validated answer found: 0 bonus
            </li>
          </ul>
        </p>
        <p>
          If you write multiple question-answer pairs <span class="red bold">(Maximum total $1.0)</span>
          <ul>
            <li>
              For each pair that passes validations: +$0.1 bonus
            </li>
            <li>
              For each pair that does not pass validations: -$0.05 bonus
            </li>
            <li>
              Bonus adjusted to be between 0 to 0.3
            </li>
            <li>
              Additionally, if all pairs pass validations and no other validated pairs found: +$0.3 bonus
            </li>
            <li>
              Additionally, if all pairs pass validations and no other validated pairs found, and you are the only one who did it: +$0.3 bonus
            </li>
          </ul>
        </p>
        <span class="large red">
          When you maintain accuracy and coverage, you will earn much more than moving on to other questions quickly!
        </span>
      `
    };
  }

  function getValidationInstructions() {
    return {
      'instructions-overview': `<p><span class="red bold">
      Please read the instructions thoroughly before beginning. Full understanding of the instructions
      will help you to retain your qualification and get <span class="hl">bonuses
      (up to 40cents per example; see Bonuses section)</span>.
    </span></p>
    <p>
      The goal of this task is to verify all possible intended question-answer pairs given a prompt question.
      For instance, given a prompt question <span class="q">When is hotel transylvania 3 going to come out?</span>,
      the possible intened question-answer pairs can be:
    </p>
    <img src="` + imgURLPrefix + `/movieRelease.png" width="550px" /><br />
    <br />
    <p>
      Other workers have written those question-answer pairs, following these criterias.
      <ol>
        <li>
          All possible intended question-answer pairs should be written. Written questions should have minimal edits from the prompt question.
        </li>
        <li>
          Written question-answer pairs should be refined to eliminate the time dependency, with up to three specific
          time-based variants, starting from December 31, 2017 and going past.
        </li>
        <li>
          If there is only one clear answer, "Single clear answer" should be marked.
        </li>
        <li>
          If the answer cannot be found in Wikipedia, or the answer cannot be expressed in a short form text, "Answer not found" should be marked.
        </li>
      </ol>
      <span class="bold">Your job is to verify whether question-answer pairs from three workers are written correctly,
      and take the union of correct pairs.</span>
    </p>
    <p>
      <span class="bold red">To fully understand the details, you should read the step-by-step instructions.
      Here is the summary of how you will work:</span>
      <ol>
        <li>
          First, assess each pair and see if it can be a intended question-answer pair. Delete the pair if it is not (e.g. the question is asking about something totally different).
          If you only see an answer with a blank question, see if it is a possible answer to the prompt question.
          Reading Wikipedia is necessary to verify whether both the question and the answer are correct---use
          the search bar to navigate Wikipedia.
        </li>
        <li>
          Among the remaining pairs, see if some answers are the same (either the texts are exactly same, or the texts are slightly different but they actually mean the same thing).
          Combine those pairs---for answers, combine texts separated by <span class="q">|</span>,
          and for questions, either choose the best one (most similar to the prompt question while differentiating other possible answers)
          or combine them separated by <span class="q">|</span>.
        </li>
        <li>
          For the blank question, write the question that is close to the prompt question
          while having edits to differentiate the given answer from other answers.
          If you find some other possible question-answer pairs, click "add pair" to add them.
          (Accurate editing will lead to the maximum bonuses.)
        </li>
        <li>
          Verify if the remaining pairs are mutually exclusive each other and written questions
          are close to the prompt question and have edits only for differentiate meanings.
          Also verify if each written question is standalone and does not depend on other written questions.
          If not, you need to edit the question.
        </li>
        <li>
          After all of these steps, if there is only one remaining pair, click "Single clear answer". If there is no remaining pairs, click "Answer not found".
        </li>
        <li>
          Submit!
        </li>
      </ol>
    </p>
    <p>
      <span class="bold">Important Notes!!! (Keep in mind before moving on to
      step-by-step instructions; necessary for bonuses)</span>
    </p>
    <p>
      <span class="bold red">Answer verification:</span>
      The answer should be <span class="bold">a phrase of
      one or multiple (usually less than 10) continuous words</span> from Wikipedia page you found (text, table or infobox).
      If there are multiple text expressions of the same answer, all the possible expressions should be written, separated by <span class="q">|</span>, as comprehensively as possible.
      If you see a sentence or explanations, edit it to be a minimal span (e.g. If the question is asking
      about the release date, <span class="q">June 13, 2018</span> instead of
      <span class="q">It was released on June 13, 2018</span>.
      Or, for some other question if the answer is <span class="q">estimated at $275 million</span>,
      write <span class="q">estimated at $275 million|$275 million</span>.)
    </p>
    <p>
      <span class="bold red">Question verification (or writing):</span>
      Written questions should be <span class="bold">as close as possible to the original question,
      only edited to the minimal extend necessary when refining the question to differentiate between multiple intentions</span>.
      This should be true even when the given question is ungrammatical or incomplete.
      Each written question should be standalone and does not depend on the other questions.
      Also, written questions should not include all questions related to the prompt
      question---each written question should be one interpretation of the prompt question,
      and each answer could be the answer to the prompt question.
      Please edit the questions if they do not meet these criteriaa.
    </p>
    <p>
      When you verify the annotations, note that multiple intentions of the question
      may only be found when you actually attmept to find the answer.
      Often the question itself seems to be clear so that you cannot think of multiple possible intentions.
      For example, the question about Hotel Transylvania 3 may look clear by itself, before actually reading
      <a href="https://en.wikipedia.org/wiki/Hotel_Transylvania_3:_Summer_Vacation">Hotel Transylvania 3: Summer Vacation</a>
      and realizing it has different film festival and theatrical release dates.
      Therefore, you should read one or multiple Wikipedia articles very carefully (at least 3-5 minutes) to find out
      <span class="red bold">what may be the user's intent when asking the question</span>.
      Note that you get <span>the maximum bonuses when your response match with other
      validators' response (see bonuses section for details)</span>.
      Each prompt question should take approximately <span class="red large">5 minutes</span>, and <span class="hl bold">you will earn much more than moving on to other questions quickly!</span>
    </p>
    <p>
      <span class="bold large">TIPS!!!</span>
      <ul>
        <li>We encourage you to use CTRL+F / CMD+F for navigating the document using keywords.</li>
        <li>More than 30% of prompt questions should have multiple question-answer pairs, and less than 5% should be "answer not found".
        If your statistics are very different from these statistics, you might have misunderstood
        the task, so please review all instructions and examples more carefully.</li>
      </ul>
    </p>
    <p><span class="bold">
      In order to see guidelines for how to correct the written question-answer pairs, see the Step-by-Step Instructions section for full details. <br />
    </span><p>
    <p><span class="red">
      You are invited as a qualified worker for validation.
      We also invite you to participate on a corresponding generation task (click my name to view it!) that is same as what you have done last week.
      Also, for the next few weeks, we will launch more batches for the same task, with 20,000 HITs in total.
      You will be able to participate if you maintain high accuracy and coverage, so please stay tuned!!
    </span></p>
      `,
      'instructions-step-by-step': `
    <p>
      Before start reading the step-by-step instructions, review the following requirements for the final response you need to submit.
      <ol>
        <li>
          All possible intended question-answer pairs should be written. Written questions should have minimal edits from the prompt question.
        </li>
        <li>
          Written question-answer pairs should be refined to eliminate the time dependency, with up to three specific
          time-based variants, starting from December 31, 2017 and going past.
        </li>
        <li>
          If there is only one clear answer, "Single clear answer" should be marked.
        </li>
        <li>
          If the answer cannot be found from Wikipedia, or the answer cannot be expressed in a short form text, "Answer not found" should be marked.
        </li>
      </ol>
    </p>
    Now, here is a step-by-step guideline.
    <br /><br />
    First, assess each pair and see if it can be a intended question-answer pair.
    <span class="bold">(Reading Wikipedia required.)</span>
    Delete the pair if it is not (e.g. the question is asking about something totally different).
    If you only see an answer with a blank question, see if it is a possible answer
    to the prompt question.
    (Note: Be <span class="bold">generous</span> in this step! If some workers thought they are answer,
    they are likely to be possible intentions. But if it is clearly incorrect, delete it.)
    If the question can be a intended question but the answer is incorrect, correct it.
    <br />
    <ul>
      <li>
        Example: <span class="q">Who won backstroke swimming in 2016 olympics?</span>
        <br />
        <img src="` + imgURLPrefix + `/swimmingWrongQuestionType2.png" width="450px" />
        ` + ARROW + `<img src="` + imgURLPrefix + `/swimmingWinner.png" width="450px" />
        <br />
        The first two pairs cannot be intended.
      </li>
    </ul>
    <ul>
      <li>
        Example: <span class="q">When did the mindy project move to hulu?</span>
        <br />
        <img src="` + imgURLPrefix + `/mindyProjectSpam.png" width="450px" /> ` + ARROW + `
        <img src="` + imgURLPrefix + `/mindyProject.png" width="350px" />
        <br />
        The first two question-answer pairs are basically same as the prompt question.
        However, the last two question-answer pairs disambiguates two different intentions of the prompt question.
        (These pairs are presumably written by a different worker.)
        Therefore, the first two pairs should be deleted and the last two pairs left.
      </li>
    </ul>

    <ul>
      <li>
        Example: <span class="q">Author of book orange is the new black?</span>
        <br />
        <img src="` + imgURLPrefix + `/bookAuthorIn.png" width="500px" />
        ` + ARROW + `<img src="` + imgURLPrefix + `/bookAuthorOut.png" width="450px" />
        <br />
        The second pair cannot be intended.
      </li>
    </ul>
    <ul>
      <li>
        Example: <span class="q">When did the mindy project move to hulu?</span>
        <br />
        <img src="` + imgURLPrefix + `/mindyProjectWrongAnswer.png" width="450px" />
        ` + ARROW + `<img src="` + imgURLPrefix + `/mindyProject.png" width="350px" />
      </li>
      <li>
        In this case, questions look good but the answer to the second question is incorrect.
        In this case we can easily correct the answer.
      </li>
    </ul>

    <br /><br />
    Now, among the remaining pairs, see if some answers are the same (either the texts are exactly same, or the texts are slightly different but they actually mean the same thing).
    Combine those pairs---for answers, combine texts separated by <span class="q">|</span>,
    and for questions, either choose the best one (most similar to the prompt question while differentiating other possible answers)
    or combine them separed by <span class="q">|</span>.
    (If the remaining pairs represent the same intention, you can simply click 'Single clear answer'
    without concerning about question.
    <br />
    <ul>
      <li>
        Example: <span class="q">The day sweden changed from left to right?</span>
        <br />
        <img src="` + imgURLPrefix + `/swedenSpam.png" width="450px" /> ` + ARROW + `
        <img src="` + imgURLPrefix + `/sweden.png" width="350px" />
        <br />
        Remaining pairs represent the same intention.
      </li>
    </ul>
    <ul>
      <li>
        Example: <span class="q">When is hotel transylvanis 3 going to come out?</span>
        <br />
        <img src="` + imgURLPrefix + `/movieReleaseIn.png" width="500px" /> ` + ARROW + `
        <img src="` + imgURLPrefix + `/movieReleaseOut.png" width="450px" />
      </li>
      <li>
        The first & third questions, and the second & fourth questions are merged.
        (Presumedly, the first & second pairs are written by one worker, and the
        third & fourth pairs are written by another worker.)
      </li>
    </ul>
    <ul>
      <li>
        <span class="q">Where is the Winter Olympics held?</span><br />
      </li>
      <li>
        <img src="` + imgURLPrefix + `/winterOlympicIn.png" width="470px" />
        ` + ARROW + `
        <img src="` + imgURLPrefix + `/winterOlympicOut.png" width="470px" />
      </li>
    </ul>

    <br /><br />
    Next, for the blank question, write the question that is close to the prompt question
    and has edits to differentaite the answer from other answers.
    Also, if you find some other possible question-answer pairs, click "add pair" to add them.
    (Accurate editing will lead to the maximum bonuses.)
    <br />
    <ul>
      <li>
        <span class="q">When did daylight saving first start in Australia?</span><br />
        <br />
        <img src="` + imgURLPrefix + `/daylightSavingIn.png" width="470px" />
          ` + ARROW + `
        <img src="` + imgURLPrefix + `/daylightSavingOut.png" width="470px" />
        <br />
        Here, all three pairs are valid, and we can write a question for the third answer.
        However, whether the answer is <span class="q">World War I</span> or <span class="q">1968</span>/<span class="q">1971</span>
        depends on whether the question intended to ask "when it ever happened" or "when has it continued".
        To reflect this ambiguity, the first and second questions are also modified.
      </li>
    </ul>
    <br /><br />
    Finally, verify if all pairs are mutually exclusive each other and questions
    are close to the prompt question while having minimal edits to differentiate meanings.
    Also, verify if each written question is standalone and does not depend on other questions.
    If not, you need to edit the question.
    <br />
    <ul class="sep">
      <li>
        Example: <span class="q">When is hotel transylvanis 3 going to come out?</span>
        <br />
        <img src="` + imgURLPrefix + `/movieReleaseTense.png" width="500px" /> ` + ARROW + `
        <img src="` + imgURLPrefix + `/movieRelease.png" width="450px" />
        <br />
        Questions were modified as the tense should be kept.
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
      `,
      'instructions-examples': `
      <ol class="sep">
        <li>
          The question have two different intents but the answer text are the same by chance.
          ` + ARROW + ` Have them to be separate question-answer pairs.
          <br />
          Example: <span class="q">Who won backstroke swimming in 2016 olympics?</span>
          <br />
          <img src="` + imgURLPrefix + `/swimmingWinner.png" width="400px" />
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
      </ol>

      `,
      'instructions-bonuses': `
        <p>
          Aside from the base pay 10c, you will get the bonuses based on validations by other crowdworkers.
          <ol>
            <li>
              If you verify the default annotations and do not make any changes, and your response matches with other validators' responses,
              you will earn 5c bonus (<span class="red bold">15c in total</span>).
            </li>
            <li>
              If you make small changes (combining answers from different text boxes, deleting question-answer pairs, etc...),
              and your response matches with othe validators' response,
              you will earn 10c (<span class="red bold">20c in total</span>).
            </li>
            <li>
              If you write or modify written questions which pass another round of validations,
              you will earn 20-30c (<span class="red bold">30-40c in total</span>).
              <ul>
                <li>20c (30c in total) when you write/modify 1-2 questions.</li>
                <li>30c (40c in total) when you write/modify 3 or more question.</li>
              </ul>
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
})();


