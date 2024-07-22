const { TeamsActivityHandler, CardFactory } = require('botbuilder');

class ProjectManagerBot extends TeamsActivityHandler {
    constructor() {
        super();
        this.projects = {};

        this.onMessage(async (context, next) => {
            const text = context.activity.text.trim().toLowerCase();
            
            if (text.startsWith('add task')) {
                await this.addTask(context);
            } else if (text.startsWith('show projects')) {
                await this.showProjects(context);
            } else if (text.startsWith('show tasks')) {
                await this.showTasks(context);
            } else {
                await context.sendActivity('I didn\'t understand. Try "add task", "show projects", or "show tasks".');
            }

            await next();
        });
    }

    async addTask(context) {
        const taskText = context.activity.text.slice(8).trim(); // Remove 'add task'
        const projectName = taskText.split(' ')[0]; // Use first word as project name
        const task = taskText.slice(projectName.length).trim();

        if (!this.projects[projectName]) {
            this.projects[projectName] = [];
        }
        this.projects[projectName].push(task);

        await context.sendActivity(`Task added to project "${projectName}": ${task}`);
    }

    async showProjects(context) {
        const projectList = Object.keys(this.projects);
        if (projectList.length === 0) {
            await context.sendActivity('No projects found.');
        } else {
            const projectsText = projectList.join(', ');
            await context.sendActivity(`Projects: ${projectsText}`);
        }
    }

    async showTasks(context) {
        const card = CardFactory.adaptiveCard(this.createProjectsAdaptiveCard());
        await context.sendActivity({ attachments: [card] });
    }

    createProjectsAdaptiveCard() {
        const card = {
            type: 'AdaptiveCard',
            body: [
                {
                    type: 'TextBlock',
                    text: 'Project Tasks',
                    weight: 'Bolder',
                    size: 'Medium'
                }
            ],
            $schema: 'http://adaptivecards.io/schemas/adaptive-card.json',
            version: '1.2'
        };

        for (const [projectName, tasks] of Object.entries(this.projects)) {
            card.body.push({
                type: 'TextBlock',
                text: projectName,
                weight: 'Bolder'
            });

            const taskList = {
                type: 'FactSet',
                facts: tasks.map((task, index) => ({
                    title: `${index + 1}.`,
                    value: task
                }))
            };

            card.body.push(taskList);
        }

        return card;
    }
}

module.exports.ProjectManagerBot = ProjectManagerBot;
