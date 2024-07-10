import {
  AxAI,
  AxBootstrapFewShot,
  AxChainOfThought, axEvalUtil,
  // axEvalUtil, type AxExample,
  type AxMetricFn
} from '../src/index.js';

interface Answer {
  summary: string
}

interface InputAnswer {
  input: string
  answer: Answer
}

const data: InputAnswer[] = [
  {
    input: `The lunar surface is covered in lunar dust and marked by mountains, impact craters, their ejecta, ray-like streaks and, mostly on the near side of the Moon, by dark maria ("seas"), which are plains of cooled magma. These maria were formed when molten lava flowed into ancient impact basins. The Moon is, except when passing through Earth's shadow during a lunar eclipse, always illuminated by the Sun, but from Earth the visible illumination shifts during its orbit, producing the lunar phases.[19] The Moon is the brightest celestial object in Earth's night sky.`,
    answer: {
      summary: 'The Moon, covered in dust and marked by mountains and craters, has dark plains formed from cooled magma and is Earth\'s brightest night sky object.'
    }
  },
  {
    input: `The Moon has been an important source of inspiration and knowledge for humans, having been crucial to cosmography, mythology, religion, art, time keeping, natural science, and spaceflight. On September 13, 1959, the first human-made object to reach an extraterrestrial body arrived on the Moon, the Soviet Union's Luna 2 impactor. In 1966, the Moon became the first extraterrestrial body where soft landings and orbital insertions were achieved. On July 20, 1969, humans for the first time landed on the Moon and any extraterrestrial body, at Mare Tranquillitatis with the lander Eagle of the United States' Apollo 11 mission. Five more crews were sent between then and 1972, each with two men landing on the surface. The longest stay was 75 hours by the Apollo 17 crew. Since then, exploration of the Moon has continued robotically, and crewed missions are being planned to return beginning in the late 2020s.`,
    answer: {
      summary: 'The Moon, crucial to human knowledge and inspiration, has been a site for significant space exploration milestones, with future missions planned.'
    }
  },
  {
    input: `Isotope dating of lunar samples suggests the Moon formed around 50 million years after the origin of the Solar System.[37][38] Historically, several formation mechanisms have been proposed,[39] but none satisfactorily explains the features of the Earth–Moon system. A fission of the Moon from Earth's crust through centrifugal force[40] would require too great an initial rotation rate of Earth.[41] Gravitational capture of a pre-formed Moon[42] depends on an unfeasibly extended atmosphere of Earth to dissipate the energy of the passing Moon.[41] A co-formation of Earth and the Moon together in the primordial accretion disk does not explain the depletion of metals in the Moon.[41] None of these hypotheses can account for the high angular momentum of the Earth–Moon system.[43]`,
    answer: {
      summary: 'Isotope dating suggests the Moon formed 50 million years post Solar System\'s origin, but no hypothesis fully explains Earth-Moon system features.'
    }
  },
  {
    input: `The Moon was volcanically active until 1.2 billion years ago, which laid down the prominent lunar maria. Most of the mare basalts erupted during the Imbrian period, 3.3–3.7 billion years ago, though some are as young as 1.2 billion years[64] and some as old as 4.2 billion years.[65] There are differing explanations for the eruption of mare basalts, particularly their uneven occurrence which mainly appear on the near-side. Causes of the distribution of the lunar highlands on the far side are also not well understood. Topological measurements show the near side crust is thinner than the far side. One possible scenario then is that large impacts on the near side may have made it easier for lava to flow onto the surface.[66]`,
    answer: {
      summary: 'The Moon was volcanically active until 1.2 billion years ago, forming lunar maria. The uneven distribution of these and lunar highlands remains unexplained.'
    }
  }
]

async function run() {
  const ai = new AxAI({
    name: 'openai',
    apiKey: process.env.apiKey ?? '',
    apiURL: process.env.apiURL ?? '',
  });

  const program = new AxChainOfThought(
    ai,
    `"Summarize this text in 20 words or less." input:string "The text to summarize" -> answer:json`,
  );


  const optimize = new AxBootstrapFewShot({
    program,
    examples: data.map((e) => ({
      input: e.input.length > 1000 ? e.input.slice(0, 500) : e.input,
      answer: e.answer,
    })),
  });

  const metricFn: AxMetricFn = ({ prediction, example }) => {
    const predictionJson = prediction as unknown as { answer: Answer };
    console.log('prediction json', predictionJson)
    const exampleJson = example as { answer: Answer };
    const scoreSummary = axEvalUtil.f1Score(
      predictionJson.answer.summary,
      exampleJson.answer.summary,
    );
    console.log('scoreSummary', scoreSummary, scoreSummary >= 0.4);
    // wasn't needed in an older version
    program.setTrace({
      answer: predictionJson.answer
    })
    console.log('program traces bbq', program.getTraces())
    return scoreSummary >= 0.4;
  };

  // Run the optimizer and save the result
  await optimize.compile(metricFn, { filename: 'output.json' });
}

run()
